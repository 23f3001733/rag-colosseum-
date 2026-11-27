"""RAG Colosseum service: POST /query  (FastAPI, no LLM = immune to prompt injection)."""
import json, math, os, re
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).parent
KB = json.loads((HERE / "knowledge_base.json").read_text())          # list of 50 strings
N = len(KB)

# ---------- tiny text helpers ----------
STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what which who whom whose when where why how many much that this these those it its as into than then
there their they you your we our i me my can could should would will shall may might must has have had
about per via over under between across use used using uses value values number amount default given
according tell please explain describe state say compare comparing difference differ differs versus vs both""".split())

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

VERSION = re.compile(r"\d+\.\d+|[a-z]+/\d")

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", text) if s.strip()]

PASS_TOKS = [set(toks(p)) for p in KB]
VOCAB = set().union(*PASS_TOKS)
SENTS = [sentences(p) for p in KB]

# ---------- indexes ----------
tfidf_full = TfidfVectorizer(analyzer=lambda t: toks(t), sublinear_tf=True)
M_full = tfidf_full.fit_transform(KB)
tfidf_head = TfidfVectorizer(analyzer=lambda t: toks(t), sublinear_tf=True)   # subject sentence only
M_head = tfidf_head.fit_transform([s[0] for s in SENTS])

DENSE = None
try:                                                                        # dense embeddings if available
    from sentence_transformers import SentenceTransformer
    DENSE = SentenceTransformer(os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    E = DENSE.encode(KB, normalize_embeddings=True)
except Exception as e:                                                      # fallback keeps service alive
    print("Dense model unavailable, using TF-IDF only:", e)

# ---------- prompt-injection scrubbing ----------
INJ = re.compile(r"ignore|disregard|forget (all|previous|prior)|previous instruction|system prompt|"
                 r"you are now|new instruction|override|reveal|jailbreak|developer mode|"
                 r"respond (only )?with|output (only )?the word|pretend|act as|</?(system|assistant)>|"
                 r"print\s+[\"']|say\s+[\"']", re.I)

def clean_question(q):
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+|\[|\]|---+", q)
    keep = [p for p in parts if p and not INJ.search(p)]
    cleaned = " ".join(keep).strip()
    return cleaned if len(cleaned) > 8 else INJ.sub(" ", q)

# ---------- retrieval ----------
def score_passages(q):
    qt = toks(q)
    q_text = " ".join(qt)
    s_full = (M_full @ tfidf_full.transform([q_text]).T).toarray().ravel()
    s_head = (M_head @ tfidf_head.transform([q_text]).T).toarray().ravel()
    lex = 0.8 * s_full + 0.2 * s_head
    vers = [t for t in qt if VERSION.fullmatch(t)]          # e.g. "1.3", "http/3": must match exactly
    for v in vers:
        for i in range(N):
            if v not in PASS_TOKS[i]:
                lex[i] *= 0.5
    if DENSE is not None:
        d = np.clip(E @ DENSE.encode([q], normalize_embeddings=True)[0], 0, 1)
        for v in vers:
            for i in range(N):
                if v not in PASS_TOKS[i]: d[i] *= 0.8
        return 0.5 * d + 0.5 * np.clip(lex * 1.6, 0, 1), d
    return np.clip(lex * 1.4, 0, 1), None

def should_abstain(q, qt, order, scores, wide=False):
    top = order[0]
    if scores[top] < 0.22 or not qt:
        return True
    # unknown proper nouns (e.g. "Elasticsearch", "Kafka") -> not in our KB
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\+\-\.]*", q)
    for i, c in enumerate(words):
        if i == 0 or not c[0].isupper() or len(c) < 3:
            continue
        for piece in toks(c):                      # "Multi-Paxos" -> multi, paxo
            if piece not in VOCAB:
                return True
    unknown = [t for t in qt if t not in VOCAB]
    if len(unknown) / len(qt) >= 0.34:
        return True
    pool = PASS_TOKS[top] | (PASS_TOKS[order[1]] if wide else set())
    cov = sum(t in pool for t in qt) / len(qt)
    return cov < 0.45

NUMQ = re.compile(r"how many|how long|how much|what port|port|timeout|threshold|interval|size|length|"
                  r"bits|ratio|rate|limit|quorum|latency|window|number|count|rate|ms\b|seconds|rtt|rto", re.I)
DF = {}
for _p in PASS_TOKS:
    for _t in _p: DF[_t] = DF.get(_t, 0) + 1

def best_sentences(q, q_tokens, pid, n=1):
    first = set(toks(SENTS[pid][0]))
    ent = {t for t in q_tokens if t in first and DF.get(t, 0) <= 3}      # subject words: already matched
    qs = (set(q_tokens) - ent) or set(q_tokens)
    numeric = bool(NUMQ.search(q))
    scored = []
    for s in SENTS[pid]:
        st = set(toks(s))
        if not st: continue
        sc = sum(1.0 / math.log(2 + DF.get(t, 1)) for t in st if t in qs) / (len(st) ** 0.3)
        if numeric and re.search(r"\d", s): sc *= 1.5
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    out = [scored[0][1]]
    for sc, s in scored[1:n]:
        if sc >= 0.4 * scored[0][0] and s not in out: out.append(s)
    return out

SYNTH = re.compile(r"\b(compare|difference|differ|versus|vs\.?|both|combined|together|relationship|"
                   r"contrast|and also|sum|total|how do .* and)\b", re.I)

class Query(BaseModel):
    question: str
    top_k: int = 3

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"ok": True, "passages": N, "dense": DENSE is not None}

@app.post("/query")
def query(body: Query):
    k = max(1, min(int(body.top_k or 3), 10))
    q = clean_question(body.question or "")
    qt = toks(q)
    scores, _ = score_passages(q)
    order = list(np.argsort(-scores))
    wide = bool(SYNTH.search(q))
    if should_abstain(q, qt, order, scores, wide):
        return {"answer": None, "sources": [], "scores": [], "confidence": round(min(0.3, float(scores[order[0]])), 4)}

    top = int(order[0])
    multi = bool(re.search(r"\band\b|\balso\b|,", q))                 # compound question -> allow 2 sentences
    ans = best_sentences(q, qt, top, n=2 if multi else 1)
    # synthesis: second passage is close in score, or question wording asks to combine/compare
    second = int(order[1])
    if SYNTH.search(q) or scores[second] >= 0.85 * scores[top] and sum(t in PASS_TOKS[second] for t in qt) / len(qt) >= 0.45:
        for s2 in best_sentences(q, qt, second):
            if s2 not in ans: ans.append(s2)
    answer = ". ".join(a.rstrip(". ") for a in ans) + "."
    src = [int(i) for i in order[:k]]
    sc = [round(float(max(0.0, min(1.0, scores[i]))), 4) for i in src]
    conf = round(0.62 + 0.38 * min(1.0, float(scores[top]) / 0.8), 4)
    return {"answer": answer, "sources": src, "scores": sc, "confidence": conf}
