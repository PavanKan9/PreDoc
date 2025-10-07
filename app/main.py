# ============================
# main.py — Part 1 of 2
# ============================
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import os, io, re, uuid, sys

# ---- OpenAI ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ---- Chroma ----
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
            print(f"[WARN] Using in-memory Chroma: {e}", file=sys.stderr)
        except Exception as e2:
            raise RuntimeError(f"Chroma init failed: {e2}")
    return c, mode

chroma_client, CHROMA_MODE = _mk_chroma()
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY or None, model_name="text-embedding-3-small"
)
COLLECTION_NAME = "shoulder_docs"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=ef, metadata={"hnsw:space": "cosine"}
)

# ---- safety / sugg fallbacks ----
try:
    from .safety import triage_flags
except Exception:
    def triage_flags(t: str): return {"blocked": False}

try:
    from .sugg import gen_suggestions
except Exception:
    def gen_suggestions(q,a,topic=None,k=3,avoid=None): return []

# ---- docx helpers ----
try:
    from .parsing import read_docx_chunks
    HAVE_READ_DOCX = True
except Exception:
    HAVE_READ_DOCX = False

def chunk_docx_bytes(data: bytes) -> List[str]:
    import docx
    doc = docx.Document(io.BytesIO(data))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    size, overlap = 1000, 150
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

# ===========================================================
# ultra-loose red-flag detector
# ===========================================================
def detect_red_flags(text: str) -> bool:
    """Trigger on any sign of pain, fever, redness, drainage, odor, swelling, opening, etc."""
    if not text: return False
    low = text.lower()
    patterns = [
        # pain & worsening
        r"pain", r"hurt", r"hurts", r"sore", r"ache", r"aching", r"burn", r"burning", r"sting",
        r"stinging", r"throb", r"sharp", r"worse", r"worsen", r"bad", r"severe", r"discomfort",
        # infection / fever
        r"fever", r"temperature", r"temp", r"chill", r"sweat", r"hot", r"heat", r"infection",
        # redness / swelling
        r"red", r"redness", r"swollen", r"swelling", r"inflamed", r"warm", r"irritation", r"discolor",
        # drainage / odor
        r"drain", r"leak", r"ooz", r"discharge", r"fluid", r"smell", r"odor", r"pus", r"wet",
        # incision / wound
        r"open", r"opening", r"split", r"separat", r"stitch", r"suture", r"incision", r"wound"
    ]
    for pat in patterns:
        if re.search(pat, low): return True
    return False

# ===========================================================
# FASTAPI
# ===========================================================
app = FastAPI(title="Patient Education")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
SESSIONS: Dict[str, Dict[str, Any]] = {}

class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ===========================================================
# /ask endpoint with red-flag insertion
# ===========================================================
@app.post("/ask")
def ask(body: AskBody):
    q_raw = (body.question or "").strip()
    if not q_raw:
        return {"answer": "Please enter a question.", "pills": [], "unverified": False}

    sid = body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"messages": []}
    SESSIONS[sid]["messages"].append({"role": "user", "content": q_raw})

    # ---- red-flag check ----
    red_flag_html = ""
    if detect_red_flags(q_raw):
        red_flag_html = (
            '<div class="bubble bot" style="border-left:4px solid #ff7a18;'
            'background:#fff4ed;color:#a64b00;">'
            '⚠️ This may be a red flag. Please contact the clinic directly: '
            'SF (415) 353-6400 / Marin (415) 886-8538'
            '</div>'
        )

    # ---- safety filter ----
    if triage_flags(q_raw).get("blocked"):
        return {"answer": "I can’t help with that request.", "pills": [], "unverified": False}

    # ---- retrieval & summarization (same as your prior version) ----
    # simplified call to your context / summarizer pipeline
    answer_text = ""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.12,
            messages=[
                {"role":"system","content":
                 "You are a medical explainer for orthopedic procedures. "
                 "Answer concisely (3–5 sentences)."},
                {"role":"user","content":q_raw}
            ])
        answer_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        answer_text = f"(Error: {e})"

    full_html = red_flag_html + f'<div class="bubble bot">{answer_text}</div>'
    SESSIONS[sid]["messages"].append({"role":"assistant","content":full_html})
    return {"answer": full_html, "pills": [], "unverified": False, "session_id": sid}

# ===========================================================
# ingest / sessions / health endpoints (same)
# ===========================================================
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        return JSONResponse({"ok": False, "error": "Upload a .docx file."}, status_code=400)
    data = await file.read()
    chunks = read_docx_chunks(file.filename) if HAVE_READ_DOCX else chunk_docx_bytes(data)
    docs, ids, metas = [], [], []
    for i, ch in enumerate(chunks):
        docs.append(ch)
        ids.append(f"{file.filename}-{i}")
        metas.append({"source": file.filename, "topic": "shoulder"})
    if docs:
        for i in range(0, len(docs), 64):
            collection.add(documents=docs[i:i+64], ids=ids[i:i+64], metadatas=metas[i:i+64])
    return {"ok": True, "chunks": len(docs)}

@app.post("/sessions/new")
def new_session():
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {"messages": []}
    return {"session_id": sid}

@app.get("/sessions/{sid}")
def read_session(sid: str):
    return {"messages": SESSIONS.get(sid, {}).get("messages", [])}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return f"ok chroma={CHROMA_MODE} llm={'ok' if client else 'missing_key'}"
# ============================
# main.py — Part 2 of 2
# ============================

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
    --accent:#0a84ff; --orange:#ff7a18; --orange-soft:#ffe8d6;
    --sidebar-w: 15rem; --sidebar-bg:#f7f7f8;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, Helvetica, Arial, sans-serif;
  }
  .app { display:grid; grid-template-columns: var(--sidebar-w) 1fr; height:100vh; width:100vw; }

  /* Sidebar */
  .sidebar { background: var(--sidebar-bg); border-right:1px solid var(--border);
    padding:16px 14px; overflow:auto; }
  .home-logo { display:flex; align-items:center; justify-content:center;
    padding:6px 4px 10px; cursor:pointer; user-select:none; }
  .home-logo img { width:100%; max-width:200px; height:auto; object-fit:contain; }
  .new-chat { display:block; width:100%; padding:10px 12px; margin-bottom:14px;
    border:1px solid var(--border); border-radius:12px; background:#fff;
    cursor:pointer; font-weight:600; }
  .side-title { font-size:13px; font-weight:600; color:#333; margin:6px 0 8px; }

  /* Main */
  .main { display:flex; flex-direction:column; min-width:0; }

  .hero { flex:1; display:flex; align-items:center; justify-content:center; padding:40px 20px; }
  .hero-inner { text-align:center; max-width:820px; }
  .hero .badge { display:inline-block; padding:6px 12px; border-radius:999px;
    background:var(--orange-soft); color:#9a4b00; font-weight:700; font-size:12px;
    letter-spacing:.12em; text-transform:uppercase; margin-bottom:14px; }
  .hero h1 { font-size:clamp(36px,4vw,52px); line-height:1.08; margin:0 0 14px; font-weight:800; }
  .hero p { color:var(--muted); margin:0 0 22px; font-size:16px; }
  .hero .selector { display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap; }
  .hero label { color:#111; font-weight:600; }
  .hero select { min-width:280px; border:2px solid var(--orange);
    border-radius:12px; padding:10px 12px; background:#fff; color:inherit; }

  .content { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  .chat-wrap { flex:1; overflow:auto; }
  .chat-col { max-width:780px; margin:0 auto; padding:18px 24px; display:flex; flex-direction:column; gap:10px; }

  .bubble { padding:12px 14px; border:1px solid var(--border); border-radius:14px;
    line-height:1.45; width:fit-content; max-width:100%; }
  .bot { background:#fafafa; align-self:flex-start; }
  .user { background:#fff; align-self:flex-end; border-color:#ddd; }

  .composer-wrap { border-top:1px solid var(--border); }
  .composer-row { max-width:780px; margin:0 auto; padding:12px 24px; }
  .composer { display:flex; align-items:center; gap:10px; width:100%;
    border:1px solid var(--border); border-radius:16px; padding:8px 12px; background:#fff; }
  .composer input { flex:1; border:none; outline:none; font-size:16px; padding:10px 12px; }
  .fab { width:42px; height:42px; border-radius:50%; background:var(--orange);
    display:flex; align-items:center; justify-content:center; cursor:pointer; border:none; }
  .fab svg { width:20px; height:20px; fill:#fff; }

  /* Orange spinner only */
  .spinner {
    width:22px; height:22px; border-radius:50%;
    border:3px solid #e6e6e6; border-top-color:var(--orange);
    animation:spin 1s linear infinite; display:inline-block;
  }
  @keyframes spin { to { transform:rotate(360deg);} }
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
  </aside>

  <main class="main">
    <section class="hero" id="hero">
      <div class="hero-inner">
        <div class="badge">Shoulder</div>
        <h1>Welcome! Select your surgery type below:</h1>
        <p>Ask questions about recovery, therapy, or precautions after surgery.</p>
        <div class="selector">
          <label for="typeHero">Type of Shoulder Arthroscopy</label>
          <select id="typeHero">
            <option>General (All Types)</option>
            <option>Rotator Cuff Repair</option>
            <option>SLAP Repair</option>
            <option>Bankart (Anterior Labrum) Repair</option>
            <option>Posterior Labrum Repair</option>
            <option>Biceps Tenodesis/Tenotomy</option>
            <option>Subacromial Decompression (SAD)</option>
            <option>Distal Clavicle Excision</option>
            <option>Capsular Release</option>
            <option>Debridement/Diagnostic Only</option>
          </select>
        </div>
      </div>
    </section>

    <div class="content" id="chatContent" style="display:flex;flex-direction:column;">
      <div class="chat-wrap"><div class="chat-col" id="chat"></div></div>

      <div class="composer-wrap">
        <div class="composer-row">
          <div class="composer">
            <input id="q" placeholder="Ask about your shoulder..." onkeydown="if(event.key==='Enter') ask()"/>
            <button class="fab" onclick="ask()" title="Send">
              <svg viewBox="0 0 24 24"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.59
              5.58L20 12l-8-8-8 8z"></path></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
let SESSION_ID = null;

function goHome(){
  document.getElementById('hero').style.display='flex';
  document.getElementById('chat').innerHTML='';
}

async function newChat(){
  const res=await fetch('/sessions/new',{method:'POST'});
  const data=await res.json(); SESSION_ID=data.session_id;
  goHome();
}

function addUser(t){
  const d=document.createElement('div');
  d.className='bubble user'; d.textContent=t;
  document.getElementById('chat').appendChild(d); scrollBottom();
}
function addBot(h){
  const d=document.createElement('div');
  d.className='bubble bot'; d.innerHTML=h;
  document.getElementById('chat').appendChild(d); scrollBottom();
}
function spinner(){
  const d=document.createElement('div');
  d.className='bubble bot';
  d.innerHTML='<span class="spinner"></span>';
  document.getElementById('chat').appendChild(d);
  scrollBottom(); return d;
}
function scrollBottom(){
  const wrap=document.querySelector('.chat-wrap');
  wrap.scrollTop=wrap.scrollHeight;
}

async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  addUser(q); document.getElementById('q').value='';
  const spin=spinner();
  const body={question:q,session_id:SESSION_ID};
  const res=await fetch('/ask',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const data=await res.json(); spin.remove();
  addBot(data.answer);
}
</script>
</body>
</html>
""")
