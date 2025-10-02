# main.py
from fastapi import FastAPI, UploadFile, File, Query
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os, io, re, uuid, sys

# ---- OpenAI (graceful if missing key) ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ---- ChromaDB with Railway-safe fallbacks ----
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.environ.get("DATA_DIR", "/data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def _mk_chroma():
    try:
        c = chromadb.PersistentClient(path=INDEX_DIR)
        mode = "persistent"
    except Exception as e:
        try:
            c = chromadb.Client()
            mode = "memory"
            print(f"[WARN] Persistent Chroma unavailable ({type(e).__name__}: {e}); using in-memory.", file=sys.stderr)
        except Exception as e2:
            print(f"[ERROR] Could not initialize Chroma at all: {e2}", file=sys.stderr)
            raise
    return c, mode

chroma_client, CHROMA_MODE = _mk_chroma()

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY or None, model_name="text-embedding-3-small"
)

COLLECTION_NAME = "shoulder_docs"
try:
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    print(f"[WARN] get_or_create_collection failed ({type(e).__name__}: {e}); retrying basic collection.", file=sys.stderr)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---- Optional safety & suggestions ----
try:
    from .safety import triage_flags
except Exception:
    def triage_flags(text: str) -> Dict[str, Any]:
        return {"blocked": False, "reasons": []}

try:
    from .sugg import gen_suggestions
except Exception:
    def gen_suggestions(q, answer, topic=None, k=4, avoid=None):
        return []

# Optional docx reader; fallback to python-docx
try:
    from .parsing import read_docx_chunks
    HAVE_READ_DOCX = True
except Exception:
    HAVE_READ_DOCX = False

def chunk_docx_bytes(file_bytes: bytes) -> List[str]:
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError("python-docx is required; add it to requirements.txt")
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    chunks, size, overlap = [], 1000, 150
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]

# ========= PROCEDURE TYPES & PILLS =========
PROCEDURE_TYPES: Dict[str, List[str]] = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff","rcr","supraspinatus","infraspinatus","subscapularis","teres minor","rc tear","cuff tear"],
    "SLAP Repair": ["slap tear","slap","superior labrum","biceps anchor","type ii slap","labral superior","slap repair"],
    "Bankart (Anterior Labrum) Repair": ["bankart","anterior labrum","anterior instability","glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum","posterior instability","reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis","tenodesis","tenotomy","biceps tendon","lhb","biceps tendinopathy"],
    "Subacromial Decompression (SAD)": ["subacromial decompression","sad","acromioplasty","impingement"],
    "Distal Clavicle Excision": ["distal clavicle excision","dce","mumford","ac joint resection","distal clavicle resection"],
    "Capsular Release": ["capsular release","adhesive capsulitis","frozen shoulder","arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement","diagnostic arthroscopy","synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

PROCEDURE_PILLS: Dict[str, List[str]] = {
    "General (All Types)": [
        "What is shoulder arthroscopy?",
        "When is it recommended?",
        "What are the risks?",
        "How long is recovery?",
    ],
    "Rotator Cuff Repair": [
        "When is rotator cuff repair recommended?",
        "How long is recovery for rotator cuff repair?",
        "What are early rehab precautions after cuff repair?",
        "What are the risks of rotator cuff repair?",
    ],
    "SLAP Repair": [
        "When is SLAP repair recommended?",
        "How long is recovery for SLAP repair?",
        "What are early rehab precautions after SLAP repair?",
        "What are the risks of SLAP repair?",
    ],
    "Bankart (Anterior Labrum) Repair": [
        "When is Bankart repair recommended?",
        "How long is recovery for Bankart repair?",
        "What instability precautions after Bankart repair?",
        "What are the risks of Bankart repair?",
    ],
    "Posterior Labrum Repair": [
        "When is posterior labrum repair recommended?",
        "How long is recovery for posterior labrum repair?",
        "What motions should I avoid early after posterior repair?",
        "What are the risks of posterior labrum repair?",
    ],
    "Biceps Tenodesis/Tenotomy": [
        "When is biceps tenodesis/tenotomy recommended?",
        "How long is recovery for biceps tenodesis/tenotomy?",
        "What are lifting precautions after tenodesis/tenotomy?",
        "What are the risks of tenodesis/tenotomy?",
    ],
    "Subacromial Decompression (SAD)": [
        "When is subacromial decompression recommended?",
        "How long is recovery for SAD?",
        "When can I return to work after SAD?",
        "What are the risks of SAD?",
    ],
    "Distal Clavicle Excision": [
        "When is distal clavicle excision recommended?",
        "How long is recovery for DCE?",
        "When can I bench press after DCE?",
        "What are the risks of DCE?",
    ],
    "Capsular Release": [
        "When is capsular release recommended?",
        "How long is recovery for capsular release?",
        "What is the PT plan after capsular release?",
        "What are the risks of capsular release?",
    ],
    "Debridement/Diagnostic Only": [
        "When is arthroscopic debridement recommended?",
        "How long is recovery after debridement?",
        "What can I do the first two weeks after debridement?",
        "What are the risks of debridement?",
    ],
}

# === Acronym & synonyms ===
ACRONYM_MAP = {
    "dce": "distal clavicle excision",
    "sad": "subacromial decompression",    # <- ensure SAD never becomes Seasonal Affective Disorder
    "rcr": "rotator cuff repair",
    "acr": "acromioplasty",
    "bt": "biceps tenodesis",
}

SYN_EXPAND = {
    "what is": ["define","explain","overview","describe"],
    "precaution": ["restriction","limit","avoid","contraindication"],
    "therapy": ["pt","physical therapy","rehab","exercises"],
    "instability": ["dislocation","subluxation"],
    "pain": ["soreness","discomfort"],
    "bench": ["press","lifting"],
}

# Add type-specific bias terms to favor correct chunks
TYPE_BIASES = {
    "Subacromial Decompression (SAD)": ["subacromial","acromion","acromioplasty","decompression","impingement","sad"],
    "Distal Clavicle Excision": ["distal clavicle","ac joint","mumford","resection","dce"],
    "Rotator Cuff Repair": ["rotator cuff","supraspinatus","infraspinatus","subscapularis","repair"],
    "SLAP Repair": ["superior labrum","biceps anchor","slap"],
    "Bankart (Anterior Labrum) Repair": ["bankart","anterior labrum","anterior instability"],
    "Posterior Labrum Repair": ["posterior labrum","posterior instability"],
    "Biceps Tenodesis/Tenotomy": ["biceps","tenodesis","tenotomy","lhb"],
    "Capsular Release": ["adhesive capsulitis","frozen shoulder","release"],
    "Debridement/Diagnostic Only": ["debridement","synovectomy","diagnostic"],
}

# ========= Content helpers =========
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician directly."
)

FORBIDDEN_IF_NOT_IN_CONTEXT = {
    "constipation","stool","stool softener","fiber","laxative","bananas",
    "soups","white bread","processed foods","dairy","gas","bloating"
}

STOPWORDS = {
    "a","an","the","and","or","to","of","for","in","on","with","after","before","is","are","be","do","does",
    "can","i","my","me","you","your","what","how","when","which","should","will","it","that","this","than"
}

def _normalize(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _tokens(s: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z]{3,}", (s or "").lower()) if w not in STOPWORDS}

def expand_query_variants(q: str, selected_type: Optional[str]) -> List[str]:
    low = q.lower().strip()

    # Expand acronyms regardless of spacing
    for a, full in ACRONYM_MAP.items():
        if re.search(rf"\b{re.escape(a)}\b", low):
            low += f" ({full})"

    variants = {q, low}

    for k, arr in SYN_EXPAND.items():
        if k in low:
            for alt in arr:
                variants.add(low.replace(k, alt))

    # add type bias into queries to force relevant retrieval
    if selected_type and selected_type in TYPE_BIASES:
        for tok in TYPE_BIASES[selected_type]:
            variants.add(low + f" {tok}")

    # one paraphrase attempt
    if client:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "Rewrite the question using common orthopedic terminology and acronyms; keep meaning the same."},
                    {"role": "user", "content": q},
                ],
            )
            p = (r.choices[0].message.content or "").strip()
            if p: variants.add(p)
        except Exception:
            pass

    return list(variants)[:8]

def _retrieve_rich(queries: List[str], n: int, topic: Optional[str], selected_type: Optional[str]):
    seen = set()
    results: List[Tuple[str, Dict[str,Any], str, float]] = []
    where: Dict[str, Any] = {}
    if topic: where["topic"] = topic
    # only pre-filter by type if specified; we’ll also re-rank
    if selected_type and selected_type != "General (All Types)":
        where_type = {"type": selected_type}
    else:
        where_type = None

    # helper to query with/without type filter
    def _do_query(q, restrict_type):
        kwargs = {"query_texts":[q], "n_results": max(8,n)}
        if topic: kwargs["where"] = {"topic": topic}
        if restrict_type and where_type:
            kwargs["where"] = {"topic": topic, "type": selected_type}
        try:
            res = collection.query(**kwargs)
        except Exception as e:
            print(f"[WARN] Retrieval failed: {e}", file=sys.stderr)
            return []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        return list(zip(docs, metas, ids))

    # try with type restriction first (if any), then without
    for q in queries:
        for restrict in (True, False):
            for d, m, _id in _do_query(q, restrict):
                if not d or not str(d).strip(): continue
                if _id in seen: continue
                seen.add(_id)
                results.append((d, m or {}, _id, 0.0))

    # simple rerank: token overlap + type bias + explicit bias terms in text
    bias_terms = set(TYPE_BIASES.get(selected_type or "", []))
    qtoks = _tokens(" ".join(queries))
    def score(item):
        d, m, _id, _ = item
        text = (d or "").lower()
        s = sum(1 for t in qtoks if t in text)
        if selected_type and (m.get("type")==selected_type): s += 3
        if bias_terms:
            s += sum(1 for t in bias_terms if t in text)
        return s
    results.sort(key=score, reverse=True)
    return [(d,m,_id) for (d,m,_id,_) in results[:max(10,n)]]

def _build_context_and_sources(pairs: List[Tuple[str,Dict[str,Any],str]], max_chars: int = 2400):
    ctx, src, total = [], [], 0
    for d,m,_id in pairs:
        nd = _normalize(d)
        if not nd: continue
        add = nd[: max(0, max_chars-total)]
        if not add: break
        ctx.append(add)
        total += len(add)
        if "source" in (m or {}): src.append(m["source"])
        if total >= max_chars: break
    context = "\n\n---\n\n".join(ctx)
    unique_sources = sorted({s for s in src if s})
    return context, unique_sources

def context_covers_question(q: str, context: str) -> bool:
    qtok = _tokens(q)
    ctx = (context or "").lower()
    hits = sum(1 for t in qtok if t in ctx)
    strong = {"recovery","risk","precaution","therapy","exercise","motion","sling","instability",
              "labrum","biceps","cuff","clavicle","acromion","decompression","tenodesis","tenotomy",
              "debridement","capsulitis","impingement","acromioplasty","subacromial"}
    return hits >= 1 or (qtok & strong)

def forbidden_if_not_in_context(answer: str, context: str) -> bool:
    a = (answer or "").lower()
    c = (context or "").lower()
    for term in FORBIDDEN_IF_NOT_IN_CONTEXT:
        if term in a and term not in c:
            return True
    return False

def _summarize_from_context(q: str, context: str) -> str:
    if not client:
        return "Server is missing OPENAI_API_KEY; please contact the clinic."
    if not context:
        return NO_MATCH_MESSAGE
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.12,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical explainer for orthopedic shoulder procedures. "
                        "Use ONLY the provided material. Provide a concise, clinically accurate answer in 3–5 sentences. "
                        "Prefer the document’s terminology; avoid speculative claims. No pleasantries. "
                        "If the material does not answer the question, reply EXACTLY with: "
                        "I couldn’t find this answered in the clinic’s provided materials. "
                        "You can try rephrasing your question, or ask your clinician directly."
                    ),
                },
                {"role": "user", "content": f"Question: {q}\n\nMaterial:\n{context}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[WARN] LLM summarize failed: {e}", file=sys.stderr)
        return NO_MATCH_MESSAGE

_BODY_PARTS = {
    "shoulder": {"shoulder"}, "knee": {"knee"}, "hip": {"hip"}, "elbow": {"elbow"},
    "wrist": {"wrist"}, "ankle": {"ankle"}, "spine": {"spine","back"}, "neck": {"neck","cervical"},
    "hand": {"hand","hands"}, "foot": {"foot","feet"},
}
def _mentioned_parts(text: str) -> set:
    low = (text or "").lower()
    return {part for part, toks in _BODY_PARTS.items() if any(t in low for t in toks)}

def _keyword_hits(txt: str, selected_type: str) -> int:
    kws = PROCEDURE_TYPES.get(selected_type, [])
    if not kws: return 0
    t = (txt or "").lower()
    return sum(1 for kw in kws if kw in t)

def adaptive_followups(last_q: str, answer: str, selected_type: str) -> List[str]:
    last = (last_q or "").lower()
    base = PROCEDURE_PILLS.get(selected_type or "General (All Types)", PROCEDURE_PILLS["General (All Types)"])
    if "recover" in last or "return" in last or "heal" in last:
        return ["What rehab milestones should I expect?","How is pain typically managed during recovery?","When can I drive again?","When do I transition from sling to full motion?"]
    if "risk" in last or "complication" in last or "safe" in last:
        return ["How are risks minimized before and after surgery?","What warning signs should prompt me to call the clinic?","How common are stiffness or re-injury?","What follow-up visits will I have?"]
    if "therapy" in last or "exercise" in last or "pt" in last:
        return ["What are the first-week exercises?","When can I start strengthening?","What motions should I avoid early on?","How often will PT sessions be?"]
    if "pain" in last or "med" in last:
        return ["How long should I expect pain after surgery?","What non-opioid options are used?","When should I taper medications?","What are red flags of uncontrolled pain?"]
    if re.search(r"\b(weeks?|months?|sling)\b", answer.lower()):
        return ["When do I start passive vs active motion?","When can I sleep without the sling?","What limits should I follow at work/school?","When can I resume sports or lifting?"]
    return base[:3]

# ========= FASTAPI =========
app = FastAPI(title="Patient Education")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ---- Static mount ----
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

# ---- Health ----
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    ok = bool(collection)
    llm = "ok" if client and OPENAI_API_KEY else "missing_key"
    return f"ok chroma={CHROMA_MODE} llm={llm}"

@app.get("/stats")
def stats():
    try:
        count = collection.count()
    except Exception as e:
        count = f"error: {e}"
    return {"collection": COLLECTION_NAME, "count": count, "chroma_mode": CHROMA_MODE}

@app.get("/debug/static", response_class=PlainTextResponse)
def debug_static():
    p = STATIC_DIR
    try:
        listing = "\n".join(sorted(f.name for f in p.glob("*"))) if p.exists() else "(dir missing)"
    except Exception as e:
        listing = f"(error listing: {e})"
    return f"STATIC_DIR={p}\nexists={p.exists()}\nfiles:\n{listing}"

# ========= UI helpers =========
@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

@app.get("/pills")
def get_pills(type: str = Query(...)):
    pills = PROCEDURE_PILLS.get(type, PROCEDURE_PILLS["General (All Types)"])
    return {"pills": pills[:3]}  # exactly 3 (grid)

# ========= Ingest =========
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        return JSONResponse({"ok": False, "error": "Please upload a .docx file."}, status_code=400)
    try:
        data = await file.read()
        if HAVE_READ_DOCX:
            path = os.path.join(DATA_DIR, file.filename)
            with open(path, "wb") as f:
                f.write(data)
            chunks = read_docx_chunks(path)
        else:
            chunks = chunk_docx_bytes(data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ingest failed: {type(e).__name__}: {e}"}, status_code=400)

    docs, ids, metas = [], [], []
    for i, ch in enumerate(chunks):
        subtype = classify_chunk(ch)
        docs.append(ch)
        ids.append(f"{file.filename}-{i}")
        metas.append({"source": file.filename, "topic": "shoulder", "type": subtype})

    if docs:
        B = 64
        for i in range(0, len(docs), B):
            collection.add(documents=docs[i:i+B], ids=ids[i:i+B], metadatas=metas[i:i+B])

    return {"ok": True, "chunks": len(docs), "file": file.filename, "chroma_mode": CHROMA_MODE}

# ========= Sessions =========
@app.get("/sessions")
def sessions():
    out = []
    for k, v in SESSIONS.items():
        title = v.get("title") or "New chat"
        if v.get("messages"):
            first = v["messages"][0]["content"]
            title = v.get("title") or (first[:40] + "…")
        out.append({"id": k, "title": title})
    out.sort(key=lambda x: x["id"], reverse=True)
    return {"sessions": out}

@app.post("/sessions/new")
def new_session():
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {"title": "New chat", "messages": [], "selected_type": None}
    return {"session_id": sid}

@app.get("/sessions/{sid}")
def read_session(sid: str):
    sess = SESSIONS.get(sid, {"messages": []})
    return {"messages": sess["messages"]}

# ========= Ask =========
@app.post("/ask")
def ask(body: AskBody):
    q_raw = (body.question or "").strip()
    if not q_raw:
        return {"answer": "Please enter a question.", "pills": [], "unverified": False}

    safety = triage_flags(q_raw)
    if safety.get("blocked"):
        return {"answer": "I can’t help with that request.", "pills": [], "unverified": False}

    sid = body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"title": "New chat", "messages": []}
    SESSIONS[sid]["messages"].append({"role": "user", "content": q_raw})
    if not SESSIONS[sid].get("title") or SESSIONS[sid]["title"] == "New chat":
        SESSIONS[sid]["title"] = q_raw[:60]

    selected_type = body.selected_type or "General (All Types)"
    SESSIONS[sid]["selected_type"] = selected_type

    # Build rich, expanded queries
    queries = expand_query_variants(q_raw, selected_type)
    pairs = _retrieve_rich(queries, n=12, topic="shoulder", selected_type=selected_type)

    # widen scope within shoulder if needed
    if len(pairs) < 3:
        widen = _retrieve_rich(queries, n=12, topic="shoulder", selected_type=None)
        seen = {p[2] for p in pairs}
        for it in widen:
            if it[2] not in seen:
                pairs.append(it); seen.add(it[2])

    context, used_sources = _build_context_and_sources(pairs, max_chars=2400)

    # Coverage check
    if not context_covers_question(q_raw, context):
        # external fallback that still respects selected type if provided
        external_answer = ""
        if client:
            try:
                type_hint = f" The user is asking in the context of {selected_type}." if selected_type else ""
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.18,
                    messages=[
                        {"role":"system","content":"Give a brief orthopedic explanation (3–5 sentences). Be accurate and neutral."},
                        {"role":"user","content": q_raw + type_hint}
                    ],
                )
                external_answer = (r.choices[0].message.content or "").strip()
            except Exception:
                external_answer = ""
        ans = external_answer or NO_MATCH_MESSAGE
        if ans != NO_MATCH_MESSAGE:
            ans += '<div style="color:#6b7280;font-size:12px;margin-top:6px;">— External reference (not clinic-verified)</div>'
        SESSIONS[sid]["messages"].append({"role":"assistant","content":ans})
        return {"answer": ans, "pills": adaptive_followups(q_raw, ans, selected_type), "unverified": True, "session_id": sid}

    answer_text = _summarize_from_context(q_raw, context)

    # body part sanity
    parts_in_q = _mentioned_parts(q_raw)
    if parts_in_q and ("shoulder" not in parts_in_q):
        ctx_low = context.lower()
        if not any(any(tok in ctx_low for tok in _BODY_PARTS[p]) for p in parts_in_q):
            answer_text = NO_MATCH_MESSAGE

    # block generic diet/constipation advice unless those terms are in context
    if forbidden_if_not_in_context(answer_text, context):
        answer_text = NO_MATCH_MESSAGE

    # If still no doc-grounded answer, try paraphrase again; else external
    if answer_text.strip() == NO_MATCH_MESSAGE.strip():
        queries2 = expand_query_variants(q_raw, selected_type)
        pairs2 = _retrieve_rich(queries2, n=12, topic="shoulder", selected_type=selected_type)
        context2, used_sources2 = _build_context_and_sources(pairs2, max_chars=2400)
        if context2 and context_covers_question(q_raw, context2):
            tmp = _summarize_from_context(q_raw, context2)
            if tmp.strip() and not forbidden_if_not_in_context(tmp, context2):
                answer_text = tmp
                used_sources = used_sources2

    verified = (answer_text.strip() != NO_MATCH_MESSAGE.strip())

    if verified and used_sources:
        src_line = "— Source: Clinic materials [" + ", ".join(used_sources[:3]) + "]"
        answer_text += f'<div style="color:#6b7280;font-size:12px;margin-top:6px;">{src_line}</div>'
    elif not verified:
        # external fallback
        external_answer = ""
        if client:
            try:
                type_hint = f" The user is asking in the context of {selected_type}." if selected_type else ""
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.18,
                    messages=[
                        {"role":"system","content":"Give a brief orthopedic explanation (3–5 sentences)."},
                        {"role":"user","content": q_raw + type_hint}
                    ],
                )
                external_answer = (r.choices[0].message.content or "").strip()
            except Exception:
                external_answer = ""
        answer_text = external_answer or NO_MATCH_MESSAGE
        if answer_text != NO_MATCH_MESSAGE:
            answer_text += '<div style="color:#6b7280;font-size:12px;margin-top:6px;">— External reference (not clinic-verified)</div>'

    try:
        sugs = gen_suggestions(q_raw, answer_text, topic="shoulder", k=3, avoid=[])
        if not sugs:
            raise ValueError("empty sugg")
        pills = [s if s.endswith("?") else s + "?" for s in sugs][:3]
    except Exception:
        pills = adaptive_followups(q_raw, answer_text, selected_type)

    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})
    return {"answer": answer_text, "pills": pills[:3], "unverified": (not verified), "session_id": sid}

# ========= UI =========
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Patient Education</title>
<style>
  :root {
    --bg:#fff; --text:#0b0b0c; --muted:#6b7280; --border:#eaeaea;
    --chip:#f6f6f6; --chip-border:#d9d9d9; --pill-border:#dbdbdb;
    --accent:#0a84ff; --orange:#ff7a18; --orange-soft:#ffe8d6;
    --sidebar-w: 15rem;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font-family: "SF Pro Text","SF Pro Display",-apple-system,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  .app { display:grid; grid-template-columns: var(--sidebar-w) 1fr; height:100vh; width:100vw; }

  /* Sidebar */
  .sidebar { border-right:1px solid var(--border); padding:16px 14px; overflow:auto; }
  .home-logo { display:flex; align-items:center; justify-content:center; padding:6px 4px 10px; cursor:pointer; user-select:none; }
  .home-logo img { width:100%; max-width: 200px; height:auto; object-fit:contain; }
  .new-chat {
    display:block; width:100%; padding:10px 12px; margin-bottom:14px;
    border:1px solid var(--border); border-radius:12px; background:#fff; cursor:pointer; font-weight:600;
  }
  .side-title { font-size:13px; font-weight:600; color:#333; margin:6px 0 8px; }
  .skeleton { height:10px; background:#f1f1f1; border-radius:8px; margin:10px 0; width:80%; }
  .skeleton:nth-child(2) { width:70%; } .skeleton:nth-child(3) { width:60%; }

  /* Main */
  .main { display:flex; flex-direction:column; min-width:0; }

  /* HERO (Welcome) */
  .hero { flex:1; display:flex; align-items:center; justify-content:center; padding:40px 20px; }
  .hero-inner { text-align:center; max-width:820px; }
  .hero .badge {
    display:inline-block; padding:6px 12px; border-radius:999px; background:var(--orange-soft); color:#9a4b00;
    font-weight:700; font-size:12px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:14px;
  }
  .hero h1 { font-size: clamp(36px, 4.6vw, 52px); line-height:1.08; margin:0 0 14px; font-weight:800; letter-spacing:-0.02em; color:#000; }
  .hero p { color:var(--muted); margin:0 0 22px; font-size:16px; }
  .hero .selector { display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap; }
  .hero label { color:#111; font-weight:600; }
  .hero select { min-width:280px; border:2px solid var(--orange); border-radius:12px; padding:10px 12px; background:#fff; color:inherit; }

  /* TOPBAR (chat view) */
  .topbar { display:none; align-items:center; justify-content:center; padding:16px 18px; border-bottom:1px solid var(--border); position:relative; }
  .title { font-size:22px; font-weight:700; letter-spacing:.2px; }
  .topic-chip {
    position:absolute; right:18px; top:12px;
    background:var(--chip); border:1px solid var(--chip-border); color:#333;
    padding:8px 14px; border-radius:999px; font-size:13px; display:flex; align-items:center; gap:8px; cursor:pointer;
  }
  .topic-panel {
    position:absolute; right:18px; top:52px; background:#fff; border:1px solid var(--border);
    border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,.06); padding:10px; display:none; z-index:10;
  }
  .topic-panel select { border:1px solid var(--border); border-radius:10px; padding:8px 10px; min-width:240px; }

  .content { flex:1; display:flex; flex-direction:column; overflow:hidden; }

  /* Chat column like ChatGPT: centered, single column */
  .chat-wrap { flex:1; overflow:auto; }
  .chat-col {
    max-width: 780px; margin: 0 auto; padding: 18px 24px;
    display:flex; flex-direction:column; gap:10px;
  }

  /* Pills: 3-column grid, full width of chat column */
  .pills {
    display:grid; grid-template-columns: repeat(3, minmax(0,1fr));
    gap:12px; padding:0; margin-bottom:8px;
  }
  .pill {
    display:inline-flex; align-items:center; justify-content:center;
    border:1px solid var(--pill-border); background:#fff; padding:12px 14px; border-radius:999px;
    font-size:15px; cursor:pointer; line-height:1.2; min-height:44px; text-align:center;
    white-space:normal; word-break:break-word;
  }

  /* Bubbles sit inside the centered column; align left/right without creating big gaps */
  .bubble {
    padding:12px 14px; border:1px solid var(--border); border-radius:14px; line-height:1.45;
    width: fit-content; max-width:100%;
  }
  .bot  { background:#fafafa; align-self:flex-start; }
  .user { background:#fff; align-self:flex-end; border-color:#ddd; }

  /* Composer locked to column width */
  .composer-wrap { border-top:1px solid var(--border); }
  .composer-row {
    max-width:780px; margin: 0 auto; padding:12px 24px;
  }
  .composer {
    display:flex; align-items:center; gap:10px; width:100%;
    border:1px solid var(--border); border-radius:16px; padding:8px 12px;
  }
  .composer input { flex:1; border:none; outline:none; font-size:16px; padding:10px 12px; }
  .fab {
    width:42px; height:42px; border-radius:50%; background:var(--orange);
    display:flex; align-items:center; justify-content:center; cursor:pointer; border:none;
  }
  .fab svg { width:20px; height:20px; fill:#fff; }

  /* Spinner: ORANGE */
  .spinner {
    width:18px; height:18px; border-radius:50%; border:3px solid #e6e6e6; border-top-color: var(--orange);
    animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:6px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="home-logo" onclick="goHome()" title="Home">
      <img src="/static/purchase-logo.png" alt="Purchase Orthopedic Clinic"/>
    </div>
    <button class="new-chat" onclick="newChat()">+ New chat</button>
    <div class="side-title">Previous Chats</div>
    <div id="chats"></div>
    <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>
  </aside>

  <main class="main">
    <!-- HERO (homescreen) -->
    <section class="hero" id="hero">
      <div class="hero-inner">
        <div class="badge">Shoulder</div>
        <h1>Welcome! Select the type of surgery below:</h1>
        <p>Choose your specific shoulder arthroscopy to tailor answers and quick questions.</p>
        <div class="selector">
          <label for="typeHero">Type of Shoulder Arthroscopy</label>
          <select id="typeHero"></select>
        </div>
      </div>
    </section>

    <!-- CHAT VIEW -->
    <div class="topbar" id="topbar">
      <div class="title">Patient Education</div>
      <div class="topic-chip" id="topicChip" onclick="toggleTopicPanel()">
        <strong>Topic:</strong> <span class="sel" id="topicText">Shoulder</span>
      </div>
      <div class="topic-panel" id="topicPanel">
        <select id="typeSelect"></select>
      </div>
    </div>

    <div class="content" id="chatContent" style="display:none;">
      <div class="chat-wrap">
        <div class="chat-col" id="chat">
          <div class="pills" id="pills"></div>
        </div>
      </div>

      <div class="composer-wrap">
        <div class="composer-row">
          <div class="composer">
            <input id="q" placeholder="Ask about your shoulder..." onkeydown="if(event.key==='Enter') ask()"/>
            <button class="fab" onclick="ask()" title="Send">
              <svg viewBox="0 0 24 24"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.59 5.58L20 12l-8-8-8 8z"></path></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
let SESSION_ID = null;
let SELECTED_TYPE = null;

function toggleTopicPanel() {
  const p = document.getElementById('topicPanel');
  p.style.display = (p.style.display === 'block') ? 'none' : 'block';
}

function goHome() {
  SELECTED_TYPE = null;
  document.getElementById('hero').style.display = 'flex';
  document.getElementById('topbar').style.display = 'none';
  document.getElementById('chatContent').style.display = 'none';
  document.getElementById('topicPanel').style.display = 'none';
  document.getElementById('chat').innerHTML = '<div class="pills" id="pills"></div>';
}

async function boot() {
  const types = await fetch('/types').then(r=>r.json()).then(d=>d.types||[]);
  const selHero = document.getElementById('typeHero');
  const selTop  = document.getElementById('typeSelect');
  [selHero, selTop].forEach(sel => {
    sel.innerHTML='';
    types.forEach(t=>{
      const o=document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o);
    });
  });
  selHero.value = "General (All Types)";
  selHero.addEventListener('change', () => handleTypeChange(selHero.value, true));
  selTop.addEventListener('change', () => handleTypeChange(selTop.value, false));

  await listSessions();
  await newChat(true);
}

function handleTypeChange(value, fromHero) {
  SELECTED_TYPE = value;
  document.getElementById('typeHero').value = value;
  document.getElementById('typeSelect').value = value;
  document.getElementById('topicText').textContent = (value==='General (All Types)') ? 'Shoulder' : value;

  document.getElementById('hero').style.display = 'none';
  document.getElementById('topbar').style.display = 'flex';
  document.getElementById('chatContent').style.display = 'flex';
  document.getElementById('topicPanel').style.display = 'none';

  const chat = document.getElementById('chat');
  chat.innerHTML = '<div class="pills" id="pills"></div>';
  renderTypePills();
  addBot('Filtering to “' + SELECTED_TYPE + '”. Ask a question or tap a pill.');
}

function renderTypePills() {
  fetch('/pills?type=' + encodeURIComponent(SELECTED_TYPE))
    .then(r=>r.json())
    .then(data => {
      const pills = data.pills || [];
      const el = document.getElementById('pills'); el.innerHTML='';
      pills.forEach(label => {
        const b = document.createElement('button'); b.className='pill'; b.textContent=label;
        b.onclick = () => { document.getElementById('q').value = label; ask(); };
        el.appendChild(b);
      });
    });
}

async function listSessions() {
  const data = await fetch('/sessions').then(r=>r.json());
  const el = document.getElementById('chats'); el.innerHTML='';
  data.sessions.forEach(s => {
    const d = document.createElement('div'); d.style.cursor='pointer'; d.style.padding='6px 2px';
    d.textContent = s.title || 'Untitled chat';
    d.onclick = () => loadSession(s.id);
    el.appendChild(d);
  });
}

async function newChat(silent=false) {
  const data = await fetch('/sessions/new', {method:'POST'}).then(r=>r.json());
  SESSION_ID = data.session_id;
  if (!silent) { goHome(); }
  await listSessions();
}

async function loadSession(id) {
  const data = await fetch('/sessions/'+id).then(r=>r.json());
  SESSION_ID = id;
  document.getElementById('hero').style.display = 'none';
  document.getElementById('topbar').style.display = 'flex';
  document.getElementById('chatContent').style.display = 'flex';
  const chat = document.getElementById('chat'); chat.innerHTML = '<div class="pills" id="pills"></div>';
  data.messages.forEach(m => { if(m.role==='user') addUser(m.content); else addBot(m.content); });
}

function addUser(text) {
  const d = document.createElement('div'); d.className='bubble user'; d.textContent=text;
  document.getElementById('chat').appendChild(d); scrollBottom();
}
function addBot(htmlText) {
  const d = document.createElement('div'); d.className='bubble bot'; d.innerHTML=htmlText;
  document.getElementById('chat').appendChild(d); scrollBottom();
}
function spinner() {
  const d = document.createElement('div'); d.className='bubble bot';
  d.innerHTML='Thinking <span class="spinner"></span>';
  document.getElementById('chat').appendChild(d); scrollBottom(); return d;
}
function scrollBottom() {
  const el = document.getElementById('chat'); el.parentElement.scrollTop = el.parentElement.scrollHeight;
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  if(!SELECTED_TYPE) { addBot('Please select a surgery type first.'); return; }
  addUser(q); document.getElementById('q').value='';
  const spin = spinner();
  const body = {question:q, session_id:SESSION_ID, selected_type:SELECTED_TYPE};
  const data = await fetch('/ask', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  }).then(r=>r.json());
  spin.remove();
  addBot(data.answer);

  if(data.pills && data.pills.length) {
    const el = document.getElementById('pills'); el.innerHTML='';
    data.pills.slice(0,3).forEach(label => {
      const b = document.createElement('button'); b.className='pill'; b.textContent=label;
      b.onclick = () => { document.getElementById('q').value = label; ask(); };
      el.appendChild(b);
    });
  }

  if(data.unverified) {
    addBot('<div style="color:#6b7280;font-size:12px;margin-top:6px;">This information is not verified by the clinic; please contact your provider with questions.</div>');
  }

  await listSessions();
}

boot();
</script>
</body>
</html>
    """)
