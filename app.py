"""RAG Colosseum service v2: POST /query (FastAPI). Extractive (no LLM) => immune to prompt injection."""
import json, math, os, re, threading, time, unicodedata
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = Path(__file__).parent
KB = json.loads((HERE / "knowledge_base.json").read_text())
N = len(KB)

# ------------------------------------------------------------------ text helpers
STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose when where why how many much that this these those it its as into than then
there their they you your we our i me my can could should would will shall may might must has have had
about per via over under between across use used using uses value values number amount default given
according tell please explain describe state say compare comparing difference differ differs versus vs both
combined total sum together altogether more fewer less larger longer shorter higher lower greater smaller
bigger add each also kindly hi hello hey quick question questions notes note wondering curious really actually basically thanks thank btw regarding concerning""".split())

def stem(w):
    """Light stemmer; endings are normalised so applies/applied/apply, scale/scales, latency/latencies all agree."""
    if w.endswith("ies") and len(w) > 5:
        w = w[:-3] + "y"
    else:
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("y"):
        w = w[:-1] + "i"
    return w

def toks(text):
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\+\./]*", text.lower().replace("-", " ")):
        w = w.rstrip("./")
        if w and w not in STOP and len(w) > 1:
            out.append(stem(w))
    return out

VERSION = re.compile(r"\d+\.\d+|[a-z]+/\d")

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", text) if s.strip()]

# abbreviations a question may use but the passages spell out
ALIASES = {
    "sgd": "stochastic gradient descent", "mvcc": "multi-version concurrency control",
    "wal": "write-ahead logging", "lsm": "log-structured merge trees", "dns": "domain name system",
    "mdns": "multicast dns", "bgp": "border gateway protocol", "ospf": "open shortest path first",
    "tcp": "transmission control protocol", "jwt": "json web tokens", "jwts": "json web tokens",
    "pkce": "proof key for code exchange", "aes": "advanced encryption standard",
    "waf": "web application firewall", "ecc": "elliptic curve cryptography",
    "hlc": "hybrid logical clocks", "2pc": "two-phase commit", "3pc": "three-phase commit",
    "mlp": "multi-layer perceptrons", "svm": "support vector machines", "cnn": "convolutional networks",
    "bn": "batch normalization", "batchnorm": "batch normalization", "layernorm": "layer normalization",
    "postgres": "postgresql", "mongo": "mongodb", "paxos": "multi-paxos", "ln": "layer normalization",
    "lr": "learning rate", "rto": "retransmission timers",
}
ALIAS_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, ALIASES)) + r")(?![A-Za-z0-9])", re.I)

SYNS = {"keepalive": "heartbeat", "keepalives": "heartbeat", "subordinate": "follower", "subordinates": "followers",
        "coordinator": "leader", "deadline": "timeout"}
SYN_RE = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(SYNS) + r")(?![A-Za-z0-9])", re.I)

def expand(q):
    extra = [ALIASES[m.group(1).lower()] for m in ALIAS_RE.finditer(q)]
    extra += [SYNS[m.group(1).lower()] for m in SYN_RE.finditer(q)]
    if re.search(r"\bhash(es|ing)?\b", q, re.I) and "digest" not in q.lower():
        extra.append("digest")
    return q + " " + " ".join(extra) if extra else q

ALIAS_STEMS = {stem(k) for k in ALIASES}

PASS_TOKS = [set(toks(p)) for p in KB]
SENTS = [sentences(p) for p in KB]
DF = {}
for _p in PASS_TOKS:
    for _t in _p:
        DF[_t] = DF.get(_t, 0) + 1
VOCAB = set(DF) | {stem(k) for k in ALIASES}
FIRST = [set(toks(s[0])) for s in SENTS]
SIG = [{t for t in FIRST[i] if DF[t] <= 3} for i in range(N)]       # "name" tokens of each passage

# ------------------------------------------------------------------ indexes
class Tfidf:
    """Tiny numpy TF-IDF (no scikit-learn => less memory on small hosts)."""
    def __init__(self, docs):
        tl = [toks(d) for d in docs]
        self.vocab = {}
        for t_ in tl:
            for t in set(t_):
                self.vocab.setdefault(t, len(self.vocab))
        df = np.zeros(len(self.vocab))
        for t_ in tl:
            for t in set(t_):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        self.M = np.vstack([self._vec(t_) for t_ in tl])
    def _vec(self, tl):
        v = np.zeros(len(self.vocab))
        for t in tl:
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1
        nz = v > 0
        v[nz] = (1 + np.log(v[nz])) * self.idf[nz]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    def sim(self, text):
        return self.M @ self._vec(toks(text))

TF = Tfidf(KB)
TH = Tfidf([s[0] for s in SENTS])
STF = Tfidf([s for ss in SENTS for s in ss])   # sentence-level keyword index

# sentence table for sentence-level semantic matching
SENT_FLAT = [s for i in range(N) for s in SENTS[i]]
OFFS, _o = [], 0
for i in range(N):
    OFFS.append((_o, _o + len(SENTS[i])))
    _o += len(SENTS[i])

ENC = None          # function: list[str] -> normalized np.ndarray
E = SE = None       # passage / sentence embeddings
ENC_NAME = "none"
_LOCK = threading.Lock()

def load_encoder():
    """Light ONNX model via fastembed (fits 512MB); falls back to sentence-transformers; else None."""
    if os.getenv("DISABLE_DENSE") == "1":
        return None, "disabled"
    name = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from fastembed import TextEmbedding
        try:
            m = TextEmbedding(model_name=name, cache_dir=str(HERE / "models"))
        except Exception:
            m = TextEmbedding(model_name=name)
        def fn(texts):
            with _LOCK:
                arr = np.array(list(m.embed(list(texts))), dtype=np.float32)
            return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        return fn, "fastembed"
    except Exception as e:
        print("fastembed unavailable:", repr(e))
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(name)
        def fn(texts):
            with _LOCK:
                return np.asarray(m.encode(list(texts), normalize_embeddings=True), dtype=np.float32)
        return fn, "sentence-transformers"
    except Exception as e:
        print("sentence-transformers unavailable:", repr(e))
    return None, "none"

def init_dense(fn, name="custom"):
    global ENC, E, SE, ENC_NAME
    E = fn(KB)
    SE = fn(SENT_FLAT)
    ENC, ENC_NAME = fn, name

try:
    _fn, _nm = load_encoder()
    if _fn is not None:
        init_dense(_fn, _nm)
except Exception as e:
    print("Dense init failed, TF-IDF only:", repr(e))
    ENC = None

# ------------------------------------------------------------------ prompt-injection scrubbing
INJ = re.compile(
    r"ignore|ignoring|disregard|forget|overrid|bypass|your (rules|programming|restrictions|constraints)|"
    r"(previous|prior|above|earlier|preceding|original) (instruction|prompt|rule|direction)|"
    r"system (prompt|message|override|instruction|note)|you are now|new (instruction|rule|task|role)|"
    r"from now on|do not (answer|follow|use|cite|search|retrieve)|don'?t (answer|follow|use)|"
    r"(always|only|just|simply|exactly) (answer|respond|reply|say|output|print|return)|"
    r"(respond|reply|answer|say|print|output|write|return|type)\s+(only\s+|just\s+|exactly\s+)?(with\b|[\"'“‘])|"
    r"the (word|phrase|string|text|token) [\"'“‘]|reveal|jailbreak|developer mode|\bdan\b|pretend|role-?play|"
    r"act as|as an ai|\bassistant:|the (correct |real |true )?answer (is|should be)|i authori[sz]e|"
    r"authori[sz]ed|sudo|\bdo anything\b|your (instructions|prompt|guidelines)|knowledge base is (wrong|outdated)|"
    r"</?(system|assistant|user|instruction)|\[/?(system|inst)|<\|", re.I)
MARK = re.compile(r"^\s*(note|system|important|admin|instruction|attention|urgent|warning|p\.?s\.?|ps|override|"
                  r"assistant|hidden|command|task)\b\s*[:\-!]", re.I)

def clean_question(q):
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"[\u200b-\u200f\u2060\ufeff\u202a-\u202e]", "", q)       # invisible characters
    q = re.sub(r"<!--.*?-->", " ", q, flags=re.S)                         # html comments
    q = re.sub(r"```.*?```", " ", q, flags=re.S)                          # code fences
    q = re.sub(r"<[^>]{1,60}>", " ", q)                                   # <system> tags
    q = re.sub(r"\[[^\]]*\]", lambda m: " " if INJ.search(m.group(0)) or MARK.match(m.group(0)[1:]) else m.group(0), q)
    parts = [p.strip() for p in re.split(r"(?<=[\.\?\!;])\s+|\n+|\s[-–—]{1,2}\s", q) if p.strip()]
    keep = []
    for p in parts:
        if re.fullmatch(r"\s*(p\.?\s?s\.?|n\.?\s?b\.?|note|important|warning|attention)\s*[\.:!]*\s*", p, re.I):
            continue
        if not (MARK.match(p) or INJ.search(p)):
            keep.append(p)
            continue
        for c in re.split(r"[,:;()]|\s[-–—]\s", p):          # clause by clause: drop only the injected clauses
            c = c.strip()
            if not c or re.fullmatch(r"(system|note|important|admin|instruction|attention|urgent|warning|override|hidden|command|task|assistant)", c, re.I):
                continue
            m = INJ.search(c)
            if not m:
                keep.append(c)
            elif len(toks(c[: m.start()])) >= 3:               # real question before the trigger words
                keep.append(c[: m.start()])
    cleaned = " ".join(keep).strip()
    return cleaned if len(toks(cleaned)) >= 2 else INJ.sub(" ", q)

# ------------------------------------------------------------------ retrieval
def score_full(text):
    qt = toks(text)
    sl = STF.sim(text)                                         # a single matching sentence is the usual signal
    slmax = np.array([sl[a:b].max() for a, b in OFFS])
    lex = 0.5 * (0.8 * TF.sim(text) + 0.2 * TH.sim(text)) + 0.5 * slmax
    pen = np.ones(N)
    for v in [t for t in qt if VERSION.fullmatch(t)]:      # "1.3", "http/3" must match exactly
        for i in range(N):
            if v not in PASS_TOKS[i]:
                pen[i] *= 0.5
    lex = lex * pen
    qv = ds = None
    if ENC is not None:
        qv = ENC([text])[0]
        ds = SE @ qv                                        # similarity to every sentence
        dp = np.clip(E @ qv, 0, 1)                          # similarity to every passage
        dmax = np.clip(np.array([ds[a:b].max() for a, b in OFFS]), 0, 1)
        d = (0.3 * dp + 0.7 * dmax) * (0.5 + 0.5 * pen)
        sc = 0.7 * d + 0.3 * np.clip(lex * 1.6, 0, 1)
    else:
        sc = np.clip(lex * 1.4, 0, 1)
    qs = set(qt)
    for i in range(N):
        if SIG[i] & qs:
            sc[i] = min(1.0, sc[i] + 0.08)                 # the question names this passage's subject
    for v in [t for t in qt if VERSION.fullmatch(t)]:      # confounders like TLS 1.2 vs 1.3
        for i in range(N):
            sc[i] = min(1.0, sc[i] + 0.12) if v in PASS_TOKS[i] else sc[i] * 0.6
    return sc, (qv, ds)

def score_vec(text):
    return score_full(text)[0]

SPLIT = re.compile(r"\s+(?:and|vs\.?|versus|or|compared\s+(?:to|with)|than|while|whereas|plus|with)\s+|\s*[,;&]\s*|\s+(?:and|or)\s*$", re.I)
SYNTH = re.compile(r"\b(compare|compared|comparison|difference|differ|differs|versus|vs\.?|both|combined|together|"
                   r"altogether|relationship|contrast|sum|total|add|plus)\b", re.I)
ARITH = re.compile(r"\b(sum|total|combined|together|altogether|add|plus|difference|how many more|how much more|"
                   r"fewer|less than|more than|larger|greater|gap|ratio|times|double|half|exceed)", re.I)

def multi_winners(q):
    """Split the question into parts; each part that names a different passage's subject gets that passage."""
    qt = toks(q)
    generic = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\+\./\-]*", q):
        ts = toks(w)
        if ts and all(DF.get(t, 0) > 3 or t not in VOCAB for t in ts):
            generic.append(w)
    gtext = " ".join(generic)
    winners = []
    for chunk in SPLIT.split(q):
        if not chunk or not chunk.strip():
            continue
        ct = set(toks(chunk))
        cands = [i for i in range(N) if SIG[i] & ct]
        if not cands:
            continue
        sc = score_vec(chunk + " " + gtext)
        best = max(cands, key=lambda i: sc[i])
        if float(sc[best]) >= 0.25 and best not in [w[0] for w in winners]:
            winners.append((best, float(sc[best])))
    return winners

def should_abstain(q, qt, best, pool, multi, numeric=False, sent_sim=None):
    if best < 0.22 or not qt:
        return True
    if ENC is not None and sent_sim is not None and sent_sim < 0.25:
        return True                                       # nothing in the KB is semantically close
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\+\-\.]*", q):
        c = m.group(0)
        before = q[: m.start()].rstrip()
        if (not before) or before[-1] in ".?!:;" or not c[0].isupper() or len(c) < 3 or c.lower() in ALIASES:
            continue                                      # sentence-initial capitals prove nothing
        if any(p not in VOCAB for p in toks(c)):          # unknown proper noun, e.g. "Elasticsearch"
            return True
    if ENC is not None and numeric and not multi:
        # "what is the <number> of ...?" paraphrased heavily: trust meaning, not shared words
        return sent_sim is None or sent_sim < 0.30
    unknown = [t for t in qt if t not in VOCAB]
    if len(unknown) / len(qt) >= (0.5 if multi else 0.34):
        return True
    cov = sum(t in pool for t in qt) / len(qt)
    return cov < (0.4 if multi else 0.45)

NUMQ = re.compile(r"how many|how long|how much|port|timeout|threshold|interval|size|length|bits|ratio|rate|limit|"
                  r"quorum|latency|window|number|count|\bms\b|seconds|rtt|lease|heartbeat|blocks|layers|batch", re.I)

def best_sentences_lex(q, q_tokens, pid, n=1):
    ent = {t for t in q_tokens if t in FIRST[pid] and DF.get(t, 0) <= 3}   # subject words already matched
    qs = (set(q_tokens) - ent) or set(q_tokens)
    numeric = bool(NUMQ.search(q))
    scored = []
    for s in SENTS[pid]:
        st = set(toks(s))
        if not st:
            continue
        sc = sum(1.0 / math.log(2 + DF.get(t, 1)) for t in st if t in qs) / (len(st) ** 0.3)
        if numeric and re.search(r"\d", s):
            sc *= 1.5
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    out = [scored[0][1]]
    for sc, s in scored[1:n]:
        if sc >= 0.4 * scored[0][0] and s not in out:
            out.append(s)
    return out

NUM_INTENT = re.compile(r"how many|how long|how much|\b(ms|milliseconds?|seconds?)\b|\bbits?\b|\bbytes?\b|\bports?\b|"
                        r"duration|interval|timeout|threshold|latency|window|coefficient|scalar|ratio|\bsize\b|\blength\b|"
                        r"\bcount\b|number of|quorum|\brate\b|limit|heartbeat|period|layers|blocks", re.I)
NOT_NUM = re.compile(r"^\s*(how (does|do|did|is|are|can|would|should)|explain|why|describe|what is the exact)", re.I)
UNITS = [
    (re.compile(r"\bms\b|millisecond|duration|latency|timeout|interval|period|delay|window|heartbeat|lease|expir", re.I), re.compile(r"\d+\s?ms\b")),
    (re.compile(r"\bbits?\b|key length|key size", re.I), re.compile(r"\d+[\s-]?bits?\b")),
    (re.compile(r"\bbytes?\b", re.I), re.compile(r"\d+\s?bytes?\b")),
    (re.compile(r"\bports?\b", re.I), re.compile(r"port\s\d+")),
    (re.compile(r"threshold|ratio|fraction|probability|saturation|utilization|coefficient|learning rate|\brate\b|burst|skew|scalar|dropout", re.I), re.compile(r"\b0\.\d+")),
]
PREM = re.compile(r"^\s*(?:since|given that|given|assuming(?: that)?|suppose(?: that)?|considering(?: that)?|knowing that|"
                  r"seeing that|because|provided that|if|as)\b[^,]*,\s*(.+)$", re.I | re.S)

PREM_FULL = re.compile(r"^\s*(?:since|given that|given|assuming(?: that)?|suppose(?: that)?|considering(?: that)?|knowing that|"
                       r"seeing that|because|provided that|if|as)\b([^,]*),\s*(.+)$", re.I | re.S)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

def premise_contradicts(base, pid):
    """True when the question plants a number (\"Since Raft needs 100 nodes...\") that the KB passage does not contain."""
    m = PREM_FULL.match(base)
    if not m:
        return False
    nums = NUM_RE.findall(m.group(1))
    if not nums:
        return False
    have = set(NUM_RE.findall(KB[pid]))
    return any(n not in have for n in nums)

def main_clause(q):
    """Drop a planted premise ("Since Raft needs 100 nodes, what is ...") and keep what is really asked."""
    m = PREM.match(q)
    return m.group(1).strip() if m and len(toks(m.group(1))) >= 2 else q

def best_sentence_single(main, q_tokens, pid, ds=None, numeric=False, n=1):
    ent = {t for t in q_tokens if t in FIRST[pid] and DF.get(t, 0) <= 3}
    qs = (set(q_tokens) - ent) or set(q_tokens)
    a, b = OFFS[pid]
    rows = []
    for k, s_ in enumerate(SENTS[pid]):
        st = set(toks(s_))
        lexs = sum(1.0 / math.log(2 + DF.get(t, 1)) for t in st if t in qs) / (len(st) ** 0.3) if st else 0.0
        rows.append([lexs, s_, 0.0])
    mx = max(r[0] for r in rows) + 1e-9
    for k, r in enumerate(rows):
        comb = r[0] / mx
        if ds is not None:
            comb = 0.7 * float(ds[a + k]) + 0.3 * comb
        if numeric:
            if re.search(r"\d", r[1]):
                comb += 0.12
            for qpat, spat in UNITS:
                if qpat.search(main) and spat.search(r[1]):
                    comb += 0.2
                    break
        r[2] = comb
    rows.sort(key=lambda r: -r[2])
    out = [rows[0][1]]
    if n > 1:
        for r in rows[1:n]:
            if r[2] >= 0.7 * rows[0][2] and r[1] not in out:
                out.append(r[1])
    return out

def pick_number(sentence, qt):
    best, best_sc = None, -1
    words = re.findall(r"[A-Za-z0-9\.\-\+/]+", sentence)
    for idx, w in enumerate(words):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|bits|bit)?", w.rstrip("."))
        if not m:
            continue
        win = toks(" ".join(words[max(0, idx - 6): idx + 4]))
        sc = sum(1 for t in win if t in qt)
        if sc > best_sc:
            best, best_sc = (float(m.group(1)), m.group(2) or ""), sc
    return best

def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"

def arithmetic_line(q, sents, qt):
    vals = [pick_number(s, set(qt)) for s in sents]
    vals = [v for v in vals if v]
    if len(vals) < 2 or not ARITH.search(q):
        return ""
    a, b = vals[0][0], vals[1][0]
    unit = vals[0][1] if vals[0][1] == vals[1][1] else ""
    line = f"Combined: {fmt(a)}{unit} + {fmt(b)}{unit} = {fmt(a + b)}{unit}; difference: {fmt(abs(a - b))}{unit}"
    if b and re.search(r"ratio|times|double|half|fold", q, re.I):
        line += f"; ratio: {a / b:.2f}"
    return line

def answer_question(q, k, base=None):
    qt = toks(q)
    base = base if base is not None else q
    main = main_clause(base)
    sq = [t for t in toks(main) if t not in ALIAS_STEMS] or [t for t in toks(base) if t not in ALIAS_STEMS] or qt
    sc, (qv, ds_full) = score_full(q)
    numeric = bool(NUM_INTENT.search(main)) and not NOT_NUM.match(main)
    order = [int(i) for i in np.argsort(-sc)]
    winners = multi_winners(q)
    multi = len(winners) >= 2
    if not multi and SYNTH.search(q) and sc[order[1]] >= 0.5 * sc[order[0]]:
        winners = [(order[0], float(sc[order[0]])), (order[1], float(sc[order[1]]))]
        multi = True
    top = order[0]
    if multi:
        pids = [w[0] for w in winners][:3]
        pool = set().union(*[PASS_TOKS[p] for p in pids])
        best = max([float(sc[top])] + [w[1] for w in winners])
    else:
        pids, pool, best = [top], PASS_TOKS[top], float(sc[top])
    if not multi and premise_contradicts(base, top):
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.2}
    ds_sel = ds_full
    if ENC is not None and main != base:
        ds_sel = SE @ ENC([expand(main)])[0]
    sent_sim = None
    if ds_sel is not None:
        a_, b_ = OFFS[top]
        sent_sim = float(ds_sel[a_:b_].max())
    if should_abstain(q, qt, best, pool, multi, numeric, sent_sim):
        return {"answer": None, "sources": [], "scores": [], "confidence": round(min(0.3, best), 4)}

    final = {i: float(sc[i]) for i in range(N)}
    for p, s in (winners if multi else []):
        final[p] = max(final[p], s)
    if multi:
        k = max(k, len(pids))
        ans, firsts = [], []
        for p in pids:
            got = best_sentences_lex(q, sq, p, n=2)
            firsts.append(got[0])
            ans += [s for s in got if s not in ans]
        extra = arithmetic_line(q, firsts, sq)
        text = ". ".join(a.rstrip(". ") for a in ans) + "."
        if extra:
            text += " " + extra + "."
    else:
        compound = bool(re.search(r"\band\b|\balso\b", main))
        ans = best_sentence_single(main, sq, top, ds_sel, numeric, n=2 if compound else 1)
        text = ". ".join(a.rstrip(". ") for a in ans) + "."
    # winners first (guaranteed in sources), then the rest by score; final list sorted descending
    chosen = list(pids) + [i for i in sorted(final, key=lambda i: -final[i]) if i not in pids]
    chosen = sorted(chosen[:k], key=lambda i: -final[i])
    scores = [round(max(0.0, min(1.0, final[i])), 4) for i in chosen]
    conf = round(0.62 + 0.38 * min(1.0, max(scores) / 0.8), 4)
    return {"answer": text, "sources": chosen, "scores": scores, "confidence": conf}

# ------------------------------------------------------------------ web app
class Query(BaseModel):
    question: str
    top_k: int = 3

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
LOG = deque(maxlen=200)
score_vec("warm up")

@app.get("/")
def health():
    return {"ok": True, "passages": N, "dense": ENC is not None, "encoder": ENC_NAME}

@app.get("/debug")
def debug():
    return list(LOG)[-60:]

@app.post("/query")
def query(body: Query):
    t0 = time.time()
    raw = body.question or ""
    k = max(1, min(int(body.top_k or 3), 10))
    try:
        base = clean_question(raw)
        cleaned = expand(base)
        res = answer_question(cleaned, k, base)
    except Exception as e:                              # never crash: abstain instead
        cleaned, res = raw, {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
        print("ERROR", repr(e))
    entry = {"mode": ENC_NAME, "q": raw[:400], "cleaned": cleaned[:300], "res": res, "ms": round((time.time() - t0) * 1000)}
    LOG.append(entry)
    print(json.dumps(entry), flush=True)
    return res
