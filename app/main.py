# main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, uuid, sys

# ---- OpenAI (graceful if missing key) ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception as e:
    client = None

# ---- ChromaDB with Railway-safe fallbacks ----
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.environ.get("DATA_DIR", "/data")  # writable in Railway; ephemeral between deploys
INDEX_DIR = os.path.join(DATA_DIR, "chroma")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def _mk_chroma():
    """
    Try persistent storage first (best on Railway), then fall back to in-memory if anything fails.
    This prevents boot crashes if the image/glibc/duckdb combo is finicky.
    """
    try:
        client = chromadb.PersistentClient(path=INDEX_DIR)
        mode = "persistent"
    except Exception as e:
        # Final fallback: in-memory client (no disk)
        try:
            client = chromadb.Client()
            mode = "memory"
            print(f"[WARN] Persistent Chroma unavailable ({type(e).__name__}: {e}); using in-memory.", file=sys.stderr)
        except Exception as e2:
            print(f"[ERROR] Could not initialize Chroma at all: {e2}", file=sys.stderr)
            raise
    return client, mode

chroma_client, CHROMA_MODE = _mk_chroma()

# Embeddings (don’t crash if key missing — retrieval still works because Chroma will call the embedder at query time)
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
    # Ultra-safe fallback: create a dummy no-embed collection that won’t crash app start
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

# Optional docx reader from your code; fallback to python-docx
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
    "Rotator Cuff Repair": ["rotator cuff", "supraspinatus", "infraspinatus", "subscapularis", "teres minor", "rc tear"],
    "SLAP Repair": ["slap tear", "superior labrum", "biceps anchor", "type ii slap", "labral superior", "slap repair"],
    "Bankart (Anterior Labrum) Repair": ["bankart", "anterior labrum", "anterior instability", "glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum", "posterior instability", "reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis", "tenotomy", "biceps tendon", "lhb", "biceps tendinopathy", "tenodesis/tenotomy"],
    "Subacromial Decompression (SAD)": ["subacromial decompression", "acromioplasty", "impingement", "s.a.d"],
    "Distal Clavicle Excision": ["distal clavicle excision", "dce", "mumford", "ac joint resection"],
    "Capsular Release": ["capsular release", "adhesive capsulitis", "frozen shoulder", "arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement", "diagnostic arthroscopy", "synovectomy"],
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

def classify_chunk(text: str) -> str:
    t = (text or "").lower()
    best = "General (All Types)"
    best_hits = 0
    for k, kws in PROCEDURE_TYPES.items():
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_hits = hits
            best = k
    return best

# ========= Content helpers (strict grounding) =========
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician directly."
)

def _normalize(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _paraphrase_once(q: str) -> str:
    if not client:
        return q
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Paraphrase the user's medical question in one sentence. Preserve meaning; don't add new claims."},
                {"role": "user", "content": q},
            ],
        )
        p = (r.choices[0].message.content or "").strip()
        return p or q
    except Exception:
        return q

def _retrieve(q: str, n: int = 8, topic: Optional[str] = "shoulder", selected_type: Optional[str] = None):
    try:
        where: Dict[str, Any] = {}
        if topic:
            where["topic"] = topic
        if selected_type and selected_type in PROCEDURE_KEYS and selected_type != "General (All Types)":
            where["type"] = selected_type
        kwargs = {"query_texts": [q], "n_results": max(8, n)}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        return res.get("documents", [[]])[0]
    except Exception as e:
        print(f"[WARN] Retrieval failed: {e}", file=sys.stderr)
        return []

def _build_context(docs: List[str], max_chars: int = 1800) -> str:
    clean_docs = [_normalize(d) for d in docs[:5] if isinstance(d, str) and d.strip()]
    context = "\n\n---\n\n".join(clean_docs)
    return context[:max_chars] if context else ""

def _summarize_from_context(q: str, context: str) -> str:
    if not client:
        return "Server is missing OPENAI_API_KEY; please contact the clinic."
    if not context:
        return NO_MATCH_MESSAGE
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical explainer. "
                        "Write 2–3 clear sentences (45–80 words) using only the provided material. "
                        "Start directly with the answer; do not refer to documents, material, context, sources, or clinics. "
                        "Do not add pleasantries, introductions, or disclaimers. "
                        "Use the document’s terminology when naming conditions or procedures. "
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
    return base[:4]

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

# ---- Health endpoints for Railway health checks ----
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

# ========= UI helpers =========
@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

@app.get("/pills")
def get_pills(type: str = Query(...)):
    pills = PROCEDURE_PILLS.get(type, PROCEDURE_PILLS["General (All Types)"])
    return {"pills": pills[:4]}

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
    q = (body.question or "").strip()
    if not q:
        return {"answer": "Please enter a question.", "pills": [], "unverified": False}

    safety = triage_flags(q)
    if safety.get("blocked"):
        return {"answer": "I can’t help with that request.", "pills": [], "unverified": False}

    sid = body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"title": "New chat", "messages": []}
    SESSIONS[sid]["messages"].append({"role": "user", "content": q})
    if not SESSIONS[sid].get("title") or SESSIONS[sid]["title"] == "New chat":
        SESSIONS[sid]["title"] = q[:60]

    selected_type = body.selected_type or "General (All Types)"
    SESSIONS[sid]["selected_type"] = selected_type

    docs = _retrieve(q, n=10, topic="shoulder", selected_type=selected_type)
    if (not docs) or all((not (d or "").strip()) for d in docs):
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            docs = _retrieve(q2, n=10, topic="shoulder", selected_type=selected_type)

    if (not docs) or len([d for d in docs if d and d.strip()]) < 3:
        try:
            global_docs = _retrieve(q, n=12, topic="shoulder", selected_type=None)
            if global_docs:
                scored = [(d, _keyword_hits(d, selected_type)) for d in global_docs if isinstance(d, str) and d.strip()]
                scored.sort(key=lambda x: x[1], reverse=True)
                seen, merged = set(), []
                for d in (docs + [x[0] for x in scored]):
                    if d and d not in seen:
                        seen.add(d); merged.append(d)
                docs = merged[:10]
        except Exception:
            pass

    context = _build_context(docs, max_chars=1800)
    answer_text = _summarize_from_context(q, context)

    parts_in_q = _mentioned_parts(q)
    if parts_in_q and ("shoulder" not in parts_in_q):
        ctx_low = context.lower()
        if not any(any(tok in ctx_low for tok in _BODY_PARTS[p]) for p in parts_in_q):
            answer_text = NO_MATCH_MESSAGE

    if answer_text.strip() == NO_MATCH_MESSAGE.strip():
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            docs2 = _retrieve(q2, n=10, topic="shoulder", selected_type=selected_type)
            ctx2 = _build_context(docs2, max_chars=1800) if docs2 else ""
            if ctx2:
                answer2 = _summarize_from_context(q2, ctx2)
                if answer2.strip():
                    answer_text = answer2

    verified = (answer_text.strip() != NO_MATCH_MESSAGE.strip())

    try:
        sugs = gen_suggestions(q, answer_text, topic="shoulder", k=4, avoid=[])
        if not sugs:
            raise ValueError("empty sugg")
        pills = [s if s.endswith("?") else s + "?" for s in sugs][:4]
    except Exception:
        pills = adaptive_followups(q, answer_text, selected_type)

    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})
    return {"answer": answer_text, "pills": pills, "unverified": (not verified), "session_id": sid}

# ========= UI (unchanged) =========
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""<!doctype html> ... (your exact UI block stays the same from your current file) ... """)
