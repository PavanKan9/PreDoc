# main.py
from fastapi import FastAPI, UploadFile, File, Query
from pathlib import Path
from fastapi.staticfiles import StaticFiles
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
except Exception:
    client = None

# ---- ChromaDB ----
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
            print(f"[WARN] Persistent Chroma unavailable: {e}", file=sys.stderr)
        except Exception as e2:
            print(f"[ERROR] Could not initialize Chroma: {e2}", file=sys.stderr)
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
except Exception:
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---- Dummy fallbacks for optional imports ----
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

try:
    from .parsing import read_docx_chunks
    HAVE_READ_DOCX = True
except Exception:
    HAVE_READ_DOCX = False

def chunk_docx_bytes(file_bytes: bytes) -> List[str]:
    import docx
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

# ========= Health =========
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

# ========= Sessions =========
@app.get("/sessions")
def sessions():
    out = []
    for k, v in SESSIONS.items():
        # Only include sessions with messages
        if not v.get("messages"):
            continue
        title = v.get("title") or "Untitled chat"
        if v["messages"]:
            first = v["messages"][0]["content"]
            title = v.get("title") or (first[:40] + "…")
        out.append({"id": k, "title": title})
    out.sort(key=lambda x: x["id"], reverse=True)
    return {"sessions": out}

@app.post("/sessions/new")
def new_session():
    sid = uuid.uuid4().hex[:10]
    # Leave title empty until first user question
    SESSIONS[sid] = {"title": "", "messages": [], "selected_type": None}
    return {"session_id": sid}

@app.get("/sessions/{sid}")
def read_session(sid: str):
    return {"messages": SESSIONS.get(sid, {"messages": []})["messages"]}

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
        SESSIONS[sid] = {"title": "", "messages": []}
    SESSIONS[sid]["messages"].append({"role": "user", "content": q})

    # Auto-title with first user question
    if not SESSIONS[sid].get("title"):
        SESSIONS[sid]["title"] = q[:60]

    # Dummy retrieval/answer
    answer_text = f"(Answering based on docs) {q}"
    pills = ["How long is recovery?", "What are the risks?", "When can I return to activity?"]

    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})
    return {"answer": answer_text, "pills": pills, "unverified": False, "session_id": sid}

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
  body { margin:0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .app { display:grid; grid-template-columns: 15rem 1fr; height:100vh; width:100vw; }
  .sidebar { background:#f7f7f8; border-right:1px solid #eaeaea; padding:16px; overflow:auto; }
  .new-chat { display:block; width:100%; padding:10px; margin-bottom:14px; border:1px solid #ccc; border-radius:12px; background:#fff; cursor:pointer; font-weight:600; }
  .main { display:flex; flex-direction:column; }
  .chat { flex:1; padding:20px; overflow:auto; }
  .bubble { padding:12px; margin:6px 0; border-radius:12px; max-width:70%; }
  .user { background:#fff; border:1px solid #ccc; margin-left:auto; }
  .bot { background:#f5f5f5; border:1px solid #ddd; }
  .composer { display:flex; border-top:1px solid #ccc; padding:10px; }
  .composer input { flex:1; border:none; outline:none; font-size:16px; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <button class="new-chat" onclick="newChat()">+ New chat</button>
    <div id="chats"></div>
  </aside>
  <main class="main">
    <div class="chat" id="chat"></div>
    <div class="composer">
      <input id="q" placeholder="Ask a question..." onkeydown="if(event.key==='Enter') ask()"/>
      <button onclick="ask()">Send</button>
    </div>
  </main>
</div>
<script>
let SESSION_ID=null;

async function listSessions() {
  const data=await fetch('/sessions').then(r=>r.json());
  const el=document.getElementById('chats'); el.innerHTML='';
  data.sessions.forEach(s=>{
    const d=document.createElement('div'); d.style.cursor='pointer'; d.textContent=s.title;
    d.onclick=()=>loadSession(s.id);
    el.appendChild(d);
  });
}

async function newChat() {
  const data=await fetch('/sessions/new',{method:'POST'}).then(r=>r.json());
  SESSION_ID=data.session_id;
  document.getElementById('chat').innerHTML='';
  await listSessions();
}

async function loadSession(id) {
  const data=await fetch('/sessions/'+id).then(r=>r.json());
  SESSION_ID=id;
  const chat=document.getElementById('chat'); chat.innerHTML='';
  data.messages.forEach(m=>{
    const d=document.createElement('div'); d.className='bubble '+(m.role==='user'?'user':'bot'); d.textContent=m.content;
    chat.appendChild(d);
  });
}

function addUser(t){const d=document.createElement('div');d.className='bubble user';d.textContent=t;document.getElementById('chat').appendChild(d);}
function addBot(t){const d=document.createElement('div');d.className='bubble bot';d.textContent=t;document.getElementById('chat').appendChild(d);}

async function ask() {
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  if(!SESSION_ID) await newChat();
  addUser(q); document.getElementById('q').value='';
  const body={question:q, session_id:SESSION_ID};
  const data=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  addBot(data.answer);
  await listSessions();
}
listSessions();
</script>
</body>
</html>
    """)
