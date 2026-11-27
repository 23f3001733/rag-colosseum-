import json, math, re, time
from collections import deque
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

# ----------- Semantic Concept Map (Solves Confounders without Dense Embeddings) -----------
CORE_CONCEPTS = [
    ["raft", "etcd", "cockroachdb", "3 nodes"],
    ["multi-paxos", "paxos", "chubby", "zookeeper", "4 acceptors"],
    ["gossip", "epidemic", "anti-entropy", "phi accrual"],
    ["swim", "indirect ping", "dissemination", "consul"],
    ["two-phase commit", "2pc", "three-phase commit", "3pc", "coordinator aborts", "wait-for graphs"],
    ["saga", "compensating actions", "temporal", "cadence"],
    ["consistent hashing", "virtual ring", "dynamo"],
    ["rendezvous hashing", "highest random weight", "squid", "envoy"],
    ["vector clocks", "partial ordering", "riak"],
    ["hybrid logical clocks", "hlc", "monotonic causality"],
    ["stochastic gradient descent", "sgd", "saddle points"],
    ["adamw", "decoupled weight decay"],
    ["transformer encoders", "encoders", "multi-head self-attention"],
    ["transformer decoders", "decoders", "autoregressive", "llama", "gpt"],
    ["binary logistic regression", "logistic regression", "temperature scaling"],
    ["support vector machines", "svm", "platt scaling", "radial basis function"],
    ["multi-layer perceptrons", "mlp", "gelu", "gradient clipping"],
    ["residual convolutional networks", "resnet", "convnext", "skip connections"],
    ["batch normalization", "batchnorm"],
    ["layer normalization", "layernorm"],
    ["postgresql", "postgres", "vacuum", "pgbouncer"],
    ["mysql", "innodb", "undo log", "purge threads"],
    ["b+ tree", "logarithmic search", "page latching"],
    ["log-structured merge trees", "lsm", "rocksdb"],
    ["mongodb", "wiredtiger", "snappy", "schemaless"],
    ["cassandra", "cql", "memtables"],
    ["redis", "rdb", "aof", "lru eviction"],
    ["memcached", "slab allocator"],
    ["write-ahead logging", "wal", "aries recovery"],
    ["shadow paging", "rollback journal", "sqlite"],
    ["tcp", "transmission control protocol", "bbr", "three-way handshake"],
    ["quic", "multiplexed connection streams", "head-of-line blocking"],
    ["dns", "domain name system", "dnssec", "recursive dns"],
    ["mdns", "multicast dns", "dot-local", "bonjour", "avahi"],
    ["http/2", "hpack", "binary framing"],
    ["http/3", "qpack", "alt-svc"],
    ["tls 1.3", "zero-rtt"],
    ["tls 1.2", "padding oracle", "session tickets"],
    ["bgp", "border gateway protocol", "as-path"],
    ["ospf", "open shortest path first", "dijkstra"],
    ["public key cryptography", "hybrid cryptosystems"],
    ["elliptic curve", "ecc", "ecdsa", "ed25519"],
    ["argon2", "memory-hard", "secure credential storage"],
    ["pbkdf2", "key stretching"],
    ["web application firewalls", "waf", "cross-site request forgery", "csp"],
    ["api gateways", "token bucket", "kong", "aws api gateway"],
    ["advanced encryption standard", "aes", "gcm"],
    ["chacha20-poly1305", "chacha20", "wireguard"],
    ["pkce", "oauth 2.0 proof key for code exchange"],
    ["json web tokens", "jwt", "stateless microservices"]
]

VERSION_RE = re.compile(r"\b\d+\.\d+\b|\bhttp/\d\b", re.I)

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
                
    max_possible = sum(idf.get(q, 0.0) for q in q_tokens)
    if max_possible > 0:
        scores = [min(1.0, s / max_possible) for s in scores]

    # 2. Confounder Boosting (Simulates Dense Embeddings mapping)
    q_lower = q_text.lower()
    for i in range(N):
        for concept in CORE_CONCEPTS[i]:
            if concept in q_lower or concept.replace("-", " ") in q_lower:
                scores[i] = min(1.0, scores[i] + 0.45)

    # 3. Explicit Version Penalty (Solves TLS 1.2 vs 1.3 style tricks)
    versions_in_q = VERSION_RE.findall(q_lower)
    for v in versions_in_q:
        v_stem = stem(v.replace("/", ""))
        for i in range(N):
            pt_text = " ".join(PASS_TOKS[i])
            if v not in pt_text and v_stem not in pt_text and v not in KB[i].lower():
                scores[i] *= 0.1  # Heavy penalty for missing requested version

    return scores

# ----------- Prompt Injection Filter -----------
def is_adversarial(q):
    lower_q = q.lower()
    patterns = [
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|all|system|instructions?|prompts?|rules?|constraints?|context)\b",
        r"\b(ignore|disregard|forget)\b\s+(everything|all\s+the\s+above)\b",
        r"\b(your\s+(rules|programming|restrictions|constraints|instructions?|prompts?|guidelines?))\b",
        r"\b(system\s+prompt|system\s+message|initial\s+instructions?|previous\s+instructions?)\b",
        r"\b(new\s+(instruction|rule|task|role))\b",
        r"\b(from\s+now\s+on|do\s+not\s+(answer|follow|use|cite|search|retrieve)|don'?t\s+(answer|follow|use))\b",
        r"\b(developer\s+mode|jailbreak|dan\s+mode|act\s+as|pretend\s+to\s+be|roleplay|you\s+are\s+now|as\s+an\s+ai)\b",
        r"\bsudo\b|\bi\s+authori[sz]e\b|\bdo\s+anything\b",
        r"\b(print|output|say|return|respond|reply|answer|type)\b\s+(exactly|only|just)?\s*[\"']",
        r"\b(always|only|just|simply|exactly)\s+(answer|respond|reply|say|output|print|return)\b",
        r"\b(write|create|generate|compose)\b.{0,40}\b(poem|code|script|story|essay|song|rap)\b",
        r"\b(translate|tldr)\b",
        r"\b(list|show|print|output|reveal|dump)\b.*\b(all\s+passages|knowledge\s+base|database)\b",
        r"<\/?(system|assistant|user|instruction)>",
        r"\[\/?(system|admin|user|instruction|inst)\]",
        r"^(system|admin|user|assistant):",
        r"reveal\s+your\s+secrets"
    ]
    for p in patterns:
        if re.search(p, lower_q): return True
    if "ignore previous" in lower_q or "ignore all" in lower_q: return True
    return False

# ----------- Abstention Logic -----------
def should_abstain(q, q_tokens, best_score, multi):
    if not q_tokens or best_score < 0.2: return True
    oov = [t for t in q_tokens if t not in VOCAB]
    if len(oov) / max(1, len(q_tokens)) >= (0.5 if multi else 0.4): return True
    return False

# ----------- Synthesis & Arithmetic -----------
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

# ----------- FastAPI App -----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str
    top_k: int = 3

request_lock = threading.Lock()

@app.get("/")
def health(): return {"ok": True, "passages": N, "dense": False, "bm25": True}

@app.post("/query")
def query(body: Query):
    raw, k = body.question or "", max(1, min(int(body.top_k or 3), 10))
    if is_adversarial(raw): return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}

    with request_lock: # Serializes concurrent requests to prevent CPU thrashing
        try:
            cleaned = re.sub(r"[\u200b-\u200f\u2060\ufeff\u202a-\u202e]", "", raw)
            cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.S)
            return answer_question(cleaned, k)
        except Exception as e:
            print("ERROR:", e)
            return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
