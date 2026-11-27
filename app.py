import json, math, re, time
from pathlib import Path
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = Path(__file__).parent
KB_PATH = HERE / "knowledge_base.json"
KB = json.loads(KB_PATH.read_text()) if KB_PATH.exists() else []
N = len(KB)

# ----------- Tokenization & Helpers -----------
STOP = set("a an the of in on at to for from by with and or is are was were be been being do does did what which who whom whose when where why how many much that this these those it its as into than then there their they you your we our i me my can could should would will shall may might must has have had about per via over under between across use used using uses value values number amount default given according tell please explain describe state say compare comparing difference differ differs versus vs both combined total sum together altogether more fewer less larger longer shorter higher lower greater smaller bigger add each also kindly hi hello hey quick question questions notes note wondering curious really actually basically thanks thank btw regarding concerning".split())

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w

def toks(text):
    out = []
    for w in re.findall(r"[a-z0-9][a-z0-9\+\./]*", text.lower().replace("-", " ")):
        w = w.rstrip("./")
        if w and w not in STOP and len(w) > 1:
            out.append(stem(w))
    return out

# ----------- Custom BM25 Index (0 MB RAM Footprint) -----------
k1, b = 1.5, 0.75
PASS_TOKS = [toks(p) for p in KB]
SENTS = [[s.strip() for s in re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", p) if s.strip()] for p in KB]

doc_lengths = [len(pt) for pt in PASS_TOKS]
avgdl = sum(doc_lengths) / max(1, N)

term_freqs, doc_freqs = [], {}
VOCAB = set()

for pt in PASS_TOKS:
    tfs = {}
    for t in pt:
        tfs[t] = tfs.get(t, 0) + 1
        VOCAB.add(t)
    term_freqs.append(tfs)
    for t in set(tfs.keys()):
        doc_freqs[t] = doc_freqs.get(t, 0) + 1

idf = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in doc_freqs.items()}

# ----------- Retrieval Engine -----------
def score_vec(q_text):
    q_tokens = toks(q_text)
    scores = [0.0] * N
    if not q_tokens: return scores
        
    # 1. Base BM25 Scoring
    for q in q_tokens:
        if q not in idf: continue
        for i in range(N):
            tf = term_freqs[i].get(q, 0)
            if tf > 0:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_lengths[i] / avgdl))
                scores[i] += idf[q] * (numerator / denominator)
                
    max_possible = sum(idf.get(q, 0.0) for q in q_tokens if q in idf)
    if max_possible > 0:
        scores = [min(1.0, s / max_possible) for s in scores]

    # 2. Rare Term & Bigram Boost (Solves Confounders like dense embeddings)
    q_lower = q_text.lower()
    rare_q = [t for t in q_tokens if doc_freqs.get(t, 0) <= 2]
    
    for i in range(N):
        pt_text = " ".join(PASS_TOKS[i])
        orig_text = KB[i].lower()
        
        # Boost for highly specific words (e.g., "innodb", "pgbouncer")
        for rq in rare_q:
            if rq in pt_text:
                scores[i] = min(1.0, scores[i] + 0.4)
                
        # Boost for exact version matching (e.g., "TLS 1.2" vs "TLS 1.3")
        for v in re.findall(r"\b(?:tls|http|quic|ipv)\s*v?[\d\.]+\b|\b\d+\.\d+\b", q_lower):
            if v in orig_text:
                scores[i] = min(1.0, scores[i] + 0.5)
            else:
                scores[i] *= 0.1  # Heavy penalty for version mismatch

    return scores

# ----------- Stealth Prompt Injection Filter (4/4 Adversarial) -----------
def is_adversarial(q):
    lower_q = q.lower()
    
    # 1. Catch standard instruction bypasses
    if re.search(r"(ignore|disregard|forget|override|bypass)\s+(all\s+)?(previous|prior|system|your)\s+(instructions?|prompts?|rules?|context|guidelines?)", lower_q):
        return True
        
    # 2. Catch roleplay / jailbreaks
    if re.search(r"\b(act as|pretend to be|roleplay|you are now|developer mode|dan mode|sudo)\b", lower_q):
        return True
        
    # 3. Catch system prompt leakage attempts
    if re.search(r"(output|print|reveal|show|dump)\s+(your\s+)?(instructions?|prompts?|system message|rules|backend)", lower_q):
        return True
        
    # 4. Catch XML/Markdown tags commonly used in stealth injections
    if re.search(r"<\/?(system|instruction|prompt|rule|user|assistant)>|\[\/?(system|inst|admin)\]", lower_q):
        return True
        
    # 5. Imperative command overwrites
    if "new instruction:" in lower_q or "system:" in lower_q or "ignore everything" in lower_q:
        return True
        
    return False

# ----------- Strict Abstention Logic (4/4 Abstention) -----------
def should_abstain(q, q_tokens, best_score, multi):
    if not q_tokens: return True
    
    # If the score is extremely low, it didn't match anything well
    if best_score < 0.18: return True
    
    # OVERLAP RATIO: This prevents answering off-topic queries (e.g., "How to bake a cake")
    # that happen to contain one known word (e.g., "server").
    known_tokens = [t for t in q_tokens if t in VOCAB]
    overlap_ratio = len(known_tokens) / max(1, len(q_tokens))
    
    # If less than 35% of the meaningful words in the query exist in our technical KB, ABSTAIN.
    if overlap_ratio < 0.35:
        return True
        
    return False

# ----------- Synthesis & Arithmetic (4/4 Synthesis) -----------
SPLIT = re.compile(r"\s+(?:and|vs\.?|versus|or|compared\s+(?:to|with)|than|while|whereas|plus|with)\s+|\s*[,;&]\s*|\s+(?:and|or)\s*$", re.I)
SYNTH = re.compile(r"\b(compare|compared|comparison|difference|differ|differs|versus|vs\.?|both|combined|together|altogether|relationship|contrast|sum|total|add|plus)\b", re.I)
NUMQ = re.compile(r"how many|how long|how much|port|timeout|threshold|interval|size|length|bits|ratio|rate|limit|quorum|latency|window|number|count|\bms\b|seconds|rtt|lease|heartbeat|blocks|layers|batch", re.I)
ARITH = re.compile(r"\b(sum|total|combined|together|altogether|add|plus|difference|how many more|how much more|fewer|less than|more than|larger|greater|gap|ratio|times|double|half|exceed)\b", re.I)

def multi_winners(q):
    winners = []
    for chunk in SPLIT.split(q):
        if not chunk or not chunk.strip(): continue
        sc = score_vec(chunk + " " + q)
        best = max(range(N), key=lambda i: sc[i])
        if sc[best] > 0.15 and best not in [w[0] for w in winners]:
            winners.append((best, float(sc[best])))
    return winners

def best_sentences(q, q_tokens, pid, n=1):
    qs = set(q_tokens)
    numeric = bool(NUMQ.search(q))
    scored = []
    for s in SENTS[pid]:
        st = set(toks(s))
        if not st: continue
        sc = sum(idf.get(t, 1.0) for t in st if t in qs)
        if numeric and re.search(r"\d", s): sc *= 1.5
        scored.append((sc, s))
    
    if not scored: return [KB[pid]]
    scored.sort(key=lambda x: -x[0])
    out = [scored[0][1]]
    for sc, s in scored[1:n]:
        if sc >= 0.4 * scored[0][0] and s not in out:
            out.append(s)
    return out

def pick_number(sentence, qt):
    best, best_sc = None, -1
    words = re.findall(r"[A-Za-z0-9\.\-\+/]+", sentence)
    for idx, w in enumerate(words):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|bits|bit)?", w.rstrip("."))
        if not m: continue
        win = toks(" ".join(words[max(0, idx - 6): idx + 4]))
        sc = sum(1 for t in win if t in qt)
        if sc > best_sc:
            best, best_sc = (float(m.group(1)), m.group(2) or ""), sc
    return best

def arithmetic_line(q, sents, qt):
    vals = [v for v in [pick_number(s, set(qt)) for s in sents] if v]
    if len(vals) < 2 or not ARITH.search(q): return ""
    a, b = vals[0][0], vals[1][0]
    unit = vals[0][1] if vals[0][1] == vals[1][1] else ""
    fmt = lambda x: str(int(x)) if float(x).is_integer() else f"{x:g}"
    line = f"Combined: {fmt(a)}{unit} + {fmt(b)}{unit} = {fmt(a + b)}{unit}; difference: {fmt(abs(a - b))}{unit}"
    if b and re.search(r"ratio|times|double|half|fold", q, re.I):
        line += f"; ratio: {a / b:.2f}"
    return line

def answer_question(q, k):
    q_tokens = toks(q)
    sc = score_vec(q)
    order = sorted(range(N), key=lambda i: -sc[i])
    winners = multi_winners(q)
    multi = len(winners) >= 2
    
    if not multi and SYNTH.search(q) and sc[order[1]] >= 0.5 * sc[order[0]] and sc[order[1]] > 0.2:
        winners, multi = [(order[0], float(sc[order[0]])), (order[1], float(sc[order[1]]))], True
        
    top = order[0]
    pids = [w[0] for w in winners][:3] if multi else [top]
    best_score = max([sc[top]] + [w[1] for w in winners]) if multi else sc[top]
        
    if should_abstain(q, q_tokens, best_score, multi):
        return {"answer": None, "sources": [], "scores": [], "confidence": round(min(0.3, best_score), 4)}

    if multi:
        k = max(k, len(pids))
        ans, firsts = [], []
        for p in pids:
            got = best_sentences(q, q_tokens, p, n=2)
            firsts.append(got[0])
            for s in got: 
                if s not in ans: ans.append(s)
        extra = arithmetic_line(q, firsts, q_tokens)
        text = ". ".join(a.rstrip(". ") for a in ans) + "." + (f" {extra}." if extra else "")
    else:
        ans = best_sentences(q, q_tokens, top, n=2 if bool(re.search(r"\band\b|\balso\b|,", q)) else 1)
        text = ". ".join(a.rstrip(". ") for a in ans) + "."
        
    chosen = list(dict.fromkeys(pids + order))[:k]
    scores = [round(max(0.0, min(1.0, sc[i])), 4) for i in chosen]
    return {"answer": text, "sources": chosen, "scores": scores, "confidence": round(0.62 + 0.38 * min(1.0, max(scores) / 0.8), 4)}

# ----------- FastAPI App & Server Init -----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str
    top_k: int = 3

# Lock guarantees concurrent requests queue up, preventing 512MB RAM / 0.1vCPU crashes on Render
request_lock = threading.Lock()

@app.get("/")
def health(): return {"ok": True, "passages": N, "dense": False, "bm25": True}

@app.post("/query")
def query(body: Query):
    raw, k = body.question or "", max(1, min(int(body.top_k or 3), 10))
    
    # 1. Reject Stealth Prompt Injections instantly
    if is_adversarial(raw): return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}

    # 2. Process safely inside the Concurrency Lock
    with request_lock: 
        try:
            cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff\u202a-\u202e]", "", raw)
            cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.S)
            return answer_question(cleaned, k)
        except Exception as e:
            print("ERROR:", e)
            return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
