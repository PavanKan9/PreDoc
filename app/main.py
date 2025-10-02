
# main.py
from fastapi import FastAPI, UploadFile, File, Query
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os, io, re, uuid, sys

# ===== OpenAI (graceful if missing key) =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ===== Safe static dir mount =====
ROOT = Path(__file__).resolve().parent
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT / "static")))
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ===== ChromaDB (Railway-safe) with fallback =====
RAILWAY = bool(os.environ.get("RAILWAY") or os.environ.get("RAILWAY_ENVIRONMENT"))
try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as e:
    chromadb = None
    embedding_functions = None
    print(f"[WARN] chromadb unavailable: {type(e).__name__}: {e}", file=sys.stderr)

class FallbackCollection:
    def __init__(self, name: str):
        self.docs: List[str] = []
        self.ids: List[str] = []
        self.metas: List[Dict[str, Any]] = []
    def add(self, documents: List[str], ids: List[str], metadatas: List[Dict[str, Any]]):
        self.docs.extend(documents); self.ids.extend(ids); self.metas.extend(metadatas)
    def count(self) -> int: return len(self.docs)
    def query(self, query_texts: List[str], n_results: int = 8, where: Optional[Dict[str, Any]] = None):
        q = (query_texts or [""])[0].lower(); where = where or {}
        scored: List[Tuple[int, str]] = []
        for d, m in zip(self.docs, self.metas):
            t = (d or "").lower(); s = 0
            for tok in set(re.findall(r"[a-z0-9]+", q)): s += 1 if tok and tok in t else 0
            for k, v in where.items():
                if str(m.get(k, "")).lower() == str(v).lower(): s += 2
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for s, d in scored if s > 0][:n_results] or self.docs[:n_results]
        return {"documents": [top_docs]}

def _mk_collection():
    force_memory = RAILWAY and (os.environ.get("ALLOW_PERSISTENT_CHROMA", "").lower() not in {"1","true","yes"})
    if chromadb and not force_memory:
        try:
            data_dir = os.environ.get("DATA_DIR", "/data")
            index_dir = os.path.join(data_dir, "chroma")
            os.makedirs(index_dir, exist_ok=True)
            client_ch = chromadb.PersistentClient(path=index_dir)
            mode = "persistent"
        except Exception as e:
            print(f"[WARN] Persistent Chroma failed: {e} -> memory", file=sys.stderr)
            try:
                client_ch = chromadb.Client(); mode = "memory"
            except Exception as e2:
                print(f"[WARN] Chroma memory failed: {e2} -> fallback", file=sys.stderr)
                return FallbackCollection("shoulder_docs"), "fallback"
        if embedding_functions and OPENAI_API_KEY:
            ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
            try:
                col = client_ch.get_or_create_collection(name="shoulder_docs", embedding_function=ef, metadata={"hnsw:space": "cosine"})
                return col, mode
            except Exception:
                col = client_ch.get_or_create_collection(name="shoulder_docs")
                return col, mode
        else:
            try:
                col = client_ch.get_or_create_collection(name="shoulder_docs")
                return col, mode
            except Exception:
                return FallbackCollection("shoulder_docs"), "fallback"
    return FallbackCollection("shoulder_docs"), "fallback"

collection, CHROMA_MODE = _mk_collection()

# ===== Optional safety & suggestions =====
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

# ===== DOCX reading =====
try:
    from .parsing import read_docx_chunks
    HAVE_READ_DOCX = True
except Exception:
    HAVE_READ_DOCX = False

def chunk_docx_bytes(file_bytes: bytes) -> List[str]:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is required; add it to requirements.txt")
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    chunks, size, overlap = [], 1000, 150
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size]); i += size - overlap
    return chunks

ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]

# ========= PROCEDURE TYPES & PILLS =========
PROCEDURE_TYPES: Dict[str, List[str]] = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff","supraspinatus","infraspinatus","subscapularis","teres minor","rc tear"],
    "SLAP Repair": ["slap tear","superior labrum","biceps anchor","type ii slap","labral superior","slap repair"],
    "Bankart (Anterior Labrum) Repair": ["bankart","anterior labrum","anterior instability","glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum","posterior instability","reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis","tenotomy","biceps tendon","lhb","biceps tendinopathy","tenodesis/tenotomy"],
    "Subacromial Decompression (SAD)": ["subacromial decompression","acromioplasty","impingement","s.a.d"],
    "Distal Clavicle Excision": ["distal clavicle excision","dce","mumford","ac joint resection"],
    "Capsular Release": ["capsular release","adhesive capsulitis","frozen shoulder","arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement","diagnostic arthroscopy","synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

PROCEDURE_PILLS: Dict[str, List[str]] = {
    "General (All Types)": ["What is shoulder arthroscopy?","When is it recommended?","What are the risks?","How long is recovery?"],
    "Rotator Cuff Repair": ["When is rotator cuff repair recommended?","How long is recovery for rotator cuff repair?","What are early rehab precautions after cuff repair?","What are the risks of rotator cuff repair?"],
    "SLAP Repair": ["When is SLAP repair recommended?","How long is recovery for SLAP repair?","What are early rehab precautions after SLAP repair?","What are the risks of SLAP repair?"],
    "Bankart (Anterior Labrum) Repair": ["When is Bankart repair recommended?","How long is recovery for Bankart repair?","What instability precautions after Bankart repair?","What are the risks of Bankart repair?"],
    "Posterior Labrum Repair": ["When is posterior labrum repair recommended?","How long is recovery for posterior labrum repair?","What motions should I avoid early after posterior repair?","What are the risks of posterior labrum repair?"],
    "Biceps Tenodesis/Tenotomy": ["When is biceps tenodesis/tenotomy recommended?","How long is recovery for biceps tenodesis/tenotomy?","What are lifting precautions after tenodesis/tenotomy?","What are the risks of tenodesis/tenotomy?"],
    "Subacromial Decompression (SAD)": ["When is subacromial decompression recommended?","How long is recovery for SAD?","When can I return to work after SAD?","What are the risks of SAD?"],
    "Distal Clavicle Excision": ["When is distal clavicle excision recommended?","How long is recovery for DCE?","When can I bench press after DCE?","What are the risks of DCE?"],
    "Capsular Release": ["When is capsular release recommended?","How long is recovery for capsular release?","What is the PT plan after capsular release?","What are the risks of capsular release?"],
    "Debridement/Diagnostic Only": ["When is arthroscopic debridement recommended?","How long is recovery after debridement?","What can I do the first two weeks after debridement?","What are the risks of debridement?"],
}

def classify_chunk(text: str) -> str:
    t = (text or "").lower()
    best, best_hits = "General (All Types)", 0
    for k, kws in PROCEDURE_TYPES.items():
        if not kws: continue
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits: best_hits, best = hits, k
    return best

NO_MATCH_MESSAGE = ("I couldn’t find this answered in the clinic’s provided materials. "
                    "You can try rephrasing your question, or ask your clinician directly.")

def _normalize(txt: str) -> str:
    if not isinstance(txt, str): return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _paraphrase_once(q: str) -> str:
    if not client: return q
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role":"system","content":"Paraphrase the user's medical question in one sentence. Preserve meaning; don't add new claims."},
                      {"role":"user","content":q}]
        )
        p = (r.choices[0].message.content or "").strip()
        return p or q
    except Exception:
        return q

def _retrieve(q: str, n: int = 8, topic: Optional[str] = "shoulder", selected_type: Optional[str] = None):
    try:
        where: Dict[str, Any] = {}
        if topic: where["topic"] = topic
        if selected_type and selected_type in PROCEDURE_KEYS and selected_type != "General (All Types)":
            where["type"] = selected_type
        kwargs = {"query_texts": [q], "n_results": max(8,n)}
        if where: kwargs["where"] = where
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
    if not client: return "Server is missing OPENAI_API_KEY; please contact the clinic."
    if not context: return NO_MATCH_MESSAGE
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.2,
            messages=[{"role":"system","content":(
                "You are a medical explainer. "
                "Write 2–3 clear sentences (45–80 words) using only the provided material. "
                "Start directly with the answer; do not refer to documents, material, context, sources, or clinics. "
                "Do not add pleasantries, introductions, or disclaimers. "
                "Use the document’s terminology when naming conditions or procedures. "
                "If the material does not answer the question, reply EXACTLY with: "
                "I couldn’t find this answered in the clinic’s provided materials. "
                "You can try rephrasing your question, or ask your clinician directly."
            )},
            {"role":"user","content":f"Question: {q}\n\nMaterial:\n{context}"}]
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[WARN] LLM summarize failed: {e}", file=sys.stderr)
        return NO_MATCH_MESSAGE

_BODY_PARTS = {
    "shoulder":{"shoulder"},"knee":{"knee"},"hip":{"hip"},"elbow":{"elbow"},
    "wrist":{"wrist"},"ankle":{"ankle"},"spine":{"spine","back"},"neck":{"neck","cervical"},
    "hand":{"hand","hands"},"foot":{"foot","feet"},
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
        return ["What rehab milestones should I expect?","How is pain typically managed during recovery?"]
    if "risk" in last or "complication" in last or "safe" in last:
        return ["How are risks minimized before and after surgery?","What warning signs should prompt me to call the clinic?"]
    if "therapy" in last or "exercise" in last or "pt" in last:
        return ["What are the first-week exercises?","When can I start strengthening?"]
    if "pain" in last or "med" in last:
        return ["How long should I expect pain after surgery?","What non-opioid options are used?"]
    return base[:2]

# ========= FASTAPI =========
app = FastAPI(title="Patient Education")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")

SESSIONS: Dict[str, Dict[str, Any]] = {}

class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ---- Health endpoints ----
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    llm = "ok" if client and OPENAI_API_KEY else "missing_key"
    try:
        _ = collection.count(); col = "ok"
    except Exception as e:
        col = f"err:{type(e).__name__}"
    return f"ok chroma={CHROMA_MODE} collection={col} llm={llm}"

@app.get("/stats")
def stats():
    try: count = collection.count()
    except Exception as e: count = f"error: {e}"
    return {"collection": "shoulder_docs", "count": count, "chroma_mode": CHROMA_MODE}

# ========= UI helpers =========
@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

@app.get("/pills")
def get_pills(type: str = Query(...)):
    pills = PROCEDURE_PILLS.get(type, PROCEDURE_PILLS["General (All Types)"])
    return {"pills": pills[:2]}  # <= TWO max

# ========= Ingest =========
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        return JSONResponse({"ok": False, "error": "Please upload a .docx file."}, status_code=400)
    try:
        data = await file.read()
        if HAVE_READ_DOCX:
            path = str(ROOT / file.filename)
            with open(path, "wb") as f: f.write(data)
            chunks = read_docx_chunks(path)
        else:
            chunks = chunk_docx_bytes(data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ingest failed: {type(e).__name__}: {e}"}, status_code=400)

    docs, ids, metas = [], [], []
    for i, ch in enumerate(chunks):
        subtype = classify_chunk(ch)
        docs.append(ch); ids.append(f"{file.filename}-{i}")
        metas.append({"source": file.filename, "topic": "shoulder", "type": subtype})

    if docs:
        B = 64
        for i in range(0, len(docs), B):
            try:
                collection.add(documents=docs[i:i+B], ids=ids[i:i+B], metadatas=metas[i:i+B])
            except Exception as e:
                print(f"[WARN] collection.add batch failed: {e}", file=sys.stderr)

    return {"ok": True, "chunks": len(docs), "file": file.filename, "chroma_mode": CHROMA_MODE}

# ========= Sessions =========
def _prune_empty_sessions():
    rm = [k for k, v in SESSIONS.items() if not v.get("messages")]
    for k in rm: del SESSIONS[k]

@app.get("/sessions")
def sessions():
    _prune_empty_sessions()
    out = []
    for k, v in SESSIONS.items():
        if not v.get("messages"):  # never expose empties
            continue
        title = v.get("title") or (v["messages"][0]["content"][:40] + "…")
        out.append({"id": k, "title": title})
    out.sort(key=lambda x: x["id"], reverse=True)
    return {"sessions": out}

@app.post("/sessions/new")
def new_session():
    # Create an empty shell; will be hidden unless a message arrives
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

    sid = body.session_id
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex[:10]
        SESSIONS[sid] = {"title": "New chat", "messages": [], "selected_type": None}

    SESSIONS[sid]["messages"].append({"role": "user", "content": q})
    if not SESSIONS[sid].get("title") or SESSIONS[sid]["title"] == "New chat":
        SESSIONS[sid]["title"] = q[:60]

    selected_type = body.selected_type or "General (All Types)"
    SESSIONS[sid]["selected_type"] = selected_type

    docs = _retrieve(q, n=10, topic="shoulder", selected_type=selected_type)
    if (not docs) or len([d for d in docs if isinstance(d, str) and d.strip()]) == 0:
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            docs = _retrieve(q2, n=10, topic="shoulder", selected_type=selected_type)

    if (not docs) or len([d for d in docs if isinstance(d, str) and d.strip()]) < 3:
        try:
            global_docs = _retrieve(q, n=12, topic="shoulder", selected_type=None)
            if global_docs:
                scored = [(d, _keyword_hits(d, selected_type)) for d in global_docs if isinstance(d, str) and d.strip()]
                scored.sort(key=lambda x: x[1], reverse=True)
                seen, merged = set(), []
                for d in (docs or []) + [x[0] for x in scored]:
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
                if answer2.strip(): answer_text = answer2

    verified = (answer_text.strip() != NO_MATCH_MESSAGE.strip())

    try:
        sugs = gen_suggestions(q, answer_text, topic="shoulder", k=2, avoid=[]) or []
        pills = [s if s.endswith("?") else s + "?" for s in sugs][:2]
    except Exception:
        pills = adaptive_followups(q, answer_text, selected_type)[:2]

    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})
    return {"answer": answer_text, "pills": pills[:2], "unverified": (not verified), "session_id": sid}

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
  body { margin:0; background:var(--bg); color:var(--text);
    font-family: "SF Pro Text","SF Pro Display",-apple-system,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .app { display:grid; grid-template-columns: var(--sidebar-w) 1fr; height:100vh; width:100vw; }

  /* Sidebar */
  .sidebar { border-right:1px solid var(--border); padding:16px 14px; overflow:auto; }
  .home-logo { display:flex; align-items:center; justify-content:center; padding:6px 4px 10px; cursor:pointer; user-select:none; }
  .home-logo img { width:100%; max-width: 200px; height:auto; object-fit:contain; }
  .new-chat { display:block; width:100%; padding:10px 12px; margin-bottom:14px; border:1px solid var(--border); border-radius:12px; background:#fff; cursor:pointer; font-weight:600; }
  .side-title { font-size:13px; font-weight:600; color:#333; margin:6px 0 8px; }
  .skeleton { height:10px; background:#f1f1f1; border-radius:8px; margin:10px 0; width:80%; }
  .skeleton:nth-child(2) { width:70%; } .skeleton:nth-child(3) { width:60%; }

  /* Main */
  .main { display:flex; flex-direction:column; min-width:0; }
  .hero { flex:1; display:flex; align-items:center; justify-content:center; padding:40px 20px; }
  .hero-inner { text-align:center; max-width:820px; }
  .hero .badge { display:inline-block; padding:6px 12px; border-radius:999px; background:var(--orange-soft); color:#9a4b00; font-weight:700; font-size:12px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:14px; }
  .hero h1 { font-size: clamp(36px, 4.6vw, 52px); line-height:1.08; margin:0 0 14px; font-weight:800; letter-spacing:-0.02em; color:#000; }
  .hero p { color:var(--muted); margin:0 0 22px; font-size:16px; }
  .hero .selector { display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap; }
  .hero label { color:#111; font-weight:600; }
  .hero select { min-width:280px; border:2px solid var(--orange); border-radius:12px; padding:10px 12px; background:#fff; color:inherit; }

  /* Topbar */
  .topbar { display:none; align-items:center; justify-content:center; padding:16px 18px; border-bottom:1px solid var(--border); position:relative; }
  .title { font-size:22px; font-weight:700; letter-spacing:.2px; }
  .topic-chip { position:absolute; right:18px; top:12px; background:var(--chip); border:1px solid var(--chip-border); color:#333; padding:8px 14px; border-radius:999px; font-size:13px; display:flex; align-items:center; gap:8px; cursor:pointer; }
  .topic-panel { position:absolute; right:18px; top:52px; background:#fff; border:1px solid var(--border); border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,.06); padding:10px; display:none; z-index:10; }
  .topic-panel select { border:1px solid var(--border); border-radius:10px; padding:8px 10px; min-width:240px; }

  /* Chat + sticky composer at bottom (ChatGPT-like) */
  .content { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  .chat-area { flex:1; overflow:auto; }
  .chat-inner { max-width: 820px; margin: 0 auto; padding: 18px 24px 96px; display:flex; flex-direction:column; gap:8px; }

  .message { display:flex; width:100%; }
  .message.bot { justify-content:flex-start; }
  .message.user { justify-content:flex-end; }
  .bubble { max-width:80%; padding:12px 14px; border:1px solid var(--border); border-radius:14px; line-height:1.45; word-wrap:break-word; word-break:break-word; }
  .bot .bubble { background:#fafafa; }
  .user .bubble { background:#fff; border-color:#ddd; }

  .composer-wrap {
    position: sticky; bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, rgba(255,255,255,0.98), rgba(255,255,255,0.92) 60%, transparent);
    border-top:1px solid var(--border);
    padding: 10px 16px 16px;
  }
  .composer-inner { max-width:820px; margin:0 auto; }

  /* Pills row INSIDE composer (never off-screen) */
  .pills { width:100%; display:grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap:10px; margin:0 0 10px; }
  .pill { display:block; border:1px solid var(--pill-border); background:#fff; padding:12px 16px; border-radius:999px; font-size:16px; cursor:pointer; text-align:center; white-space:normal; }

  .composer {
    display:flex; align-items:center; gap:10px;
    border:1px solid var(--border); border-radius:16px; padding:8px 12px; background:#fff;
  }
  .composer input { flex:1; border:none; outline:none; font-size:16px; padding:10px 12px; }
  .fab { width:42px; height:42px; border-radius:50%; background:#ff7a18; display:flex; align-items:center; justify-content:center; cursor:pointer; border:none; }
  .fab svg { width:20px; height:20px; fill:#fff; }

  .spinner { width:18px; height:18px; border-radius:50%; border:3px solid #e6e6e6; border-top-color:#ff7a18; animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }

  @media (max-width:520px) {
    .bubble { max-width:100%; }
    .pills { grid-template-columns: 1fr; }
  }
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
    <!-- HERO -->
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

    <!-- CHAT -->
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
      <div class="chat-area"><div class="chat-inner" id="chat"></div></div>

      <!-- Sticky bottom like ChatGPT: pills + input -->
      <div class="composer-wrap">
        <div class="composer-inner">
          <div class="pills" id="pills"></div>
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
  document.getElementById('chat').innerHTML = '';
  document.getElementById('pills').innerHTML = '';
  const selHero = document.getElementById('typeHero');
  if (selHero && selHero.options.length) selHero.value = "General (All Types)";
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
  // Do NOT pre-create a session; we only create on first message
}

function handleTypeChange(value, fromHero) {
  SELECTED_TYPE = value;
  document.getElementById('typeHero').value = value;
  document.getElementById('typeSelect').value = value;
  document.getElementById('topicText').textContent = (value==='General (All Types)') ? 'Shoulder' : value;

  document.getElementById('hero').style.display = 'none';
  document.getElementById('topbar').style.display = 'flex';
  document.getElementById('chatContent').style.display = 'block';
  document.getElementById('topicPanel').style.display = 'none';

  document.getElementById('chat').innerHTML = '';
  renderTypePills();
  addBot('Filtering to “' + SELECTED_TYPE + '”. Ask a question or tap a quick question.');
}

function renderTypePills() {
  fetch('/pills?type=' + encodeURIComponent(SELECTED_TYPE))
    .then(r=>r.json())
    .then(data => {
      const pills = (data.pills || []).slice(0,2);
      const el = document.getElementById('pills'); el.innerHTML='';
      pills.forEach(label => {
        const b = document.createElement('button'); b.className='pill'; b.textContent=label;
        b.onclick = () => { document.getElementById('q').value = label; ask(); };
        el.appendChild(b);
      });
    });
}

async function listSessions() {
  await fetch('/sessions').then(r=>r.json()).then(data => {
    const el = document.getElementById('chats'); el.innerHTML='';
    (data.sessions || []).forEach(s => {
      const d = document.createElement('div'); d.style.cursor='pointer'; d.style.padding='6px 2px';
      d.textContent = s.title || 'Untitled chat';
      d.onclick = () => loadSession(s.id);
      el.appendChild(d);
    });
  });
}

function newChat() {
  // Just reset UI (ChatGPT behavior). Do not create a session yet.
  SESSION_ID = null;
  goHome();
  listSessions();
}

async function loadSession(id) {
  const data = await fetch('/sessions/'+id).then(r=>r.json());
  SESSION_ID = id;
  const chat = document.getElementById('chat'); chat.innerHTML='';
  document.getElementById('hero').style.display = 'none';
  document.getElementById('topbar').style.display = 'flex';
  document.getElementById('chatContent').style.display = 'block';
  data.messages.forEach(m => { if(m.role==='user') addUser(m.content); else addBot(m.content); });
  renderTypePills();
}

function addUser(text) {
  const row = document.createElement('div'); row.className='message user';
  const b = document.createElement('div'); b.className='bubble'; b.textContent=text;
  row.appendChild(b);
  document.getElementById('chat').appendChild(row);
  scrollBottom();
}
function addBot(htmlText) {
  const row = document.createElement('div'); row.className='message bot';
  const b = document.createElement('div'); b.className='bubble'; b.innerHTML=htmlText;
  row.appendChild(b);
  document.getElementById('chat').appendChild(row);
  scrollBottom();
}
function spinner() {
  const row = document.createElement('div'); row.className='message bot';
  const b = document.createElement('div'); b.className='bubble';
  b.innerHTML='Thinking <span class="spinner"></span>';
  row.appendChild(b);
  document.getElementById('chat').appendChild(row);
  scrollBottom();
  return row;
}
function scrollBottom() {
  const area = document.querySelector('.chat-area');
  area.scrollTop = area.scrollHeight;
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  if(!SELECTED_TYPE) { addBot('Please select a surgery type first.'); return; }

  addUser(q);
  document.getElementById('q').value='';
  const spinRow = spinner();

  const body = {question:q, session_id:SESSION_ID, selected_type:SELECTED_TYPE};
  const data = await fetch('/ask', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  }).then(r=>r.json());

  spinRow.remove();
  addBot(data.answer);
  if (!SESSION_ID && data.session_id) SESSION_ID = data.session_id; // create on first ask

  const el = document.getElementById('pills'); el.innerHTML='';
  (data.pills || []).slice(0,2).forEach(label => {
    const b = document.createElement('button'); b.className='pill'; b.textContent=label;
    b.onclick = () => { document.getElementById('q').value = label; ask(); };
    el.appendChild(b);
  });

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
