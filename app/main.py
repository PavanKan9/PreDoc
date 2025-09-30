# main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, uuid

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---- Project-local safety (no-op fallback) ----
try:
    from .safety import triage_flags
except Exception:
    def triage_flags(text: str) -> Dict[str, Any]:
        return {"blocked": False, "reasons": []}

# ========= ENV & GLOBALS =========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "chroma"), exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma"))
COLLECTION_NAME = "shoulder_arthroscopy"
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# ========= SHOULDER ARTHROSCOPY SUBTYPES =========
PROCEDURE_TYPES = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff", "supraspinatus", "infraspinatus", "subscapularis", "teres minor", "rc tear"],
    "SLAP Repair": ["slap tear", "superior labrum", "biceps anchor", "type ii slap", "labral superior", "slap repair"],
    "Bankart (Anterior Labrum) Repair": ["bankart", "anterior labrum", "anterior instability", "glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum", "posterior instability", "reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis", "tenotomy", "biceps tendon", "lhb"],
    "Subacromial Decompression (SAD)": ["subacromial decompression", "acromioplasty", "impingement", "s.a.d"],
    "Distal Clavicle Excision": ["distal clavicle excision", "dce", "mumford", "ac joint resection"],
    "Capsular Release": ["capsular release", "adhesive capsulitis", "frozen shoulder", "arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement", "diagnostic arthroscopy", "synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

# Type-specific starter pills
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
    t = text.lower()
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

# ========= HELPERS =========
def chunk_docx(file_bytes: bytes) -> List[str]:
    try:
        import docx
    except ImportError:
        raise RuntimeError("python-docx is required. pip install python-docx")
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    chunks, size, overlap = [], 1000, 150
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def make_llm_answer(prompt: str, system: str = "You are a concise medical explainer.") -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=320,
    )
    return resp.choices[0].message.content.strip()

def summarize_to_2_or_3_sentences(text: str) -> str:
    return make_llm_answer(
        "Summarize the answer into 2–3 patient-friendly sentences without losing key specifics:\n\n" + text,
        system="You compress long answers into 2–3 clear sentences."
    )

def adaptive_followups(last_q: str, answer: str, selected_type: str) -> List[str]:
    last = (last_q or "").lower()
    base = PROCEDURE_PILLS.get(selected_type or "General (All Types)", PROCEDURE_PILLS["General (All Types)"])
    if "recover" in last or "return" in last or "heal" in last:
        return [
            "What rehab milestones should I expect?",
            "How is pain typically managed during recovery?",
            "When can I drive again?",
            "When do I transition from sling to full motion?",
        ]
    if "risk" in last or "complication" in last or "safe" in last:
        return [
            "How are risks minimized before and after surgery?",
            "What warning signs should prompt me to call the clinic?",
            "How common are stiffness or re-injury?",
            "What follow-up visits will I have?",
        ]
    if "pt" in last or "therapy" in last or "exercise" in last:
        return [
            "What are the first-week exercises?",
            "When can I start strengthening?",
            "What motions should I avoid early on?",
            "How often will PT sessions be?",
        ]
    if "pain" in last or "med" in last or "block" in last:
        return [
            "How long should I expect pain after surgery?",
            "What non-opioid options are used?",
            "When should I taper medications?",
            "What are red flags of uncontrolled pain?",
        ]
    if re.search(r"\b(weeks?|months?)\b", answer.lower()) or "sling" in answer.lower():
        return [
            "When do I start passive vs active motion?",
            "When can I sleep without the sling?",
            "What limits should I follow at work/school?",
            "When can I resume sports or lifting?",
        ]
    return base[:4]

def likely_covered(question: str, docs: List[str], dists: List[float], dist_thresh: float = 0.35) -> bool:
    # 1) vector distance; 2) keyword overlap (captures paraphrases)
    if not dists or not docs:
        return False
    if min(dists) <= dist_thresh:
        return True
    q_tokens = set(re.findall(r"[a-z]{3,}", question.lower()))
    for d in docs[:3]:
        d_tokens = set(re.findall(r"[a-z]{3,}", d.lower()))
        if len(q_tokens & d_tokens) >= 3:
            return True
    return False

# ========= FASTAPI APP =========
app = FastAPI(title="Patient Education")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory sessions
SESSIONS: Dict[str, Dict[str, Any]] = {}  # {id: {title, messages: [...], selected_type}}

# ========= MODELS =========
class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ========= ROUTES =========
@app.get("/", response_class=HTMLResponse)
def home():
    # IMPORTANT: plain triple-quoted string (NOT an f-string) to avoid brace-escaping issues.
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
    font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  }
  .app { display:grid; grid-template-columns: var(--sidebar-w) 1fr; height:100vh; width:100vw; }

  /* Sidebar */
  .sidebar { border-right:1px solid var(--border); padding:16px 14px; overflow:auto; }
  .new-chat {
    display:block; width:100%; padding:10px 12px; margin-bottom:14px;
    border:1px solid var(--border); border-radius:12px; background:#fff; cursor:pointer; font-weight:600;
  }
  .side-title { font-size:13px; font-weight:600; color:#333; margin:6px 0 8px; }
  .skeleton { height:10px; background:#f1f1f1; border-radius:8px; margin:10px 0; width:80%; }
  .skeleton:nth-child(2) { width:70%; } .skeleton:nth-child(3) { width:60%; }

  /* Main */
  .main { display:flex; flex-direction:column; min-width:0; }

  /* HERO */
  .hero { flex:1; display:flex; align-items:center; justify-content:center; padding:40px 20px; }
  .hero-inner { text-align:center; max-width:820px; }
  .hero .badge {
    display:inline-block; padding:6px 12px; border-radius:999px; background:var(--orange-soft); color:#9a4b00;
    font-weight:700; font-size:12px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:14px;
  }
  .hero h1 {
    font-size: clamp(30px, 4.5vw, 46px);
    line-height:1.08; margin:0 0 10px; font-weight:800; letter-spacing:-0.02em;
    background: linear-gradient(90deg, var(--text), #333 60%, var(--orange));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color:var(--muted); margin:0 0 22px; font-size:16px; }
  .hero .selector { display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap; }
  .hero label { color:#111; font-weight:600; }
  .hero select { min-width:280px; border:1px solid var(--border); border-radius:12px; padding:10px 12px; }

  /* TOPBAR (chat view) */
  .topbar { display:none; align-items:center; justify-content:center; padding:16px 18px; border-bottom:1px solid var(--border); position:relative; }
  .title { font-size:22px; font-weight:700; letter-spacing:.2px; }
  .topic-chip {
    position:absolute; right:18px; top:12px;
    background:var(--chip); border:1px solid var(--chip-border); color:#333;
    padding:8px 14px; border-radius:999px; font-size:13px;
    display:flex; align-items:center; gap:8px; cursor:pointer;
  }
  .topic-panel {
    position:absolute; right:18px; top:52px; background:#fff; border:1px solid var(--border);
    border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,.06); padding:10px; display:none; z-index:10;
  }
  .topic-panel select { border:1px solid var(--border); border-radius:10px; padding:8px 10px; min-width:240px; }

  .content { flex:1; display:flex; flex-direction:column; overflow:hidden; }
  .chat-area { flex:1; overflow:auto; padding:18px 24px; }
  .pills { display:flex; flex-wrap:wrap; gap:14px; padding:0 24px 12px; }
  .pill {
    border:1px solid var(--pill-border); background:#fff; padding:12px 16px; border-radius:999px;
    font-size:16px; cursor:pointer; line-height:1; box-shadow: 0 1px 0 rgba(0,0,0,0.02);
  }

  .bubble { max-width:820px; padding:12px 14px; border:1px solid var(--border); border-radius:14px; margin:8px 0; line-height:1.45; }
  .bot { background:#fafafa; }
  .user { background:#fff; margin-left:auto; border-color:#ddd; }

  .composer-wrap { border-top:1px solid var(--border); padding:12px 24px; }
  .composer {
    display:flex; align-items:center; gap:10px; max-width:920px;
    border:1px solid var(--border); border-radius:16px; padding:8px 12px; margin:0 auto;
  }
  .composer input { flex:1; border:none; outline:none; font-size:16px; padding:10px 12px; }
  .fab {
    width:42px; height:42px; border-radius:50%; background:var(--orange);
    display:flex; align-items:center; justify-content:center; cursor:pointer; border:none;
  }
  .fab svg { width:20px; height:20px; fill:#fff; }

  .spinner {
    width:18px; height:18px; border-radius:50%; border:3px solid #e6e6e6; border-top-color:#0a84ff;
    animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:6px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
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

    <!-- CHAT VIEW (after selection) -->
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
      <div class="chat-area" id="chat"></div>
      <div class="pills" id="pills"></div>

      <div class="composer-wrap">
        <div class="composer">
          <input id="q" placeholder="Ask about your shoulder..." onkeydown="if(event.key==='Enter') ask()"/>
          <button class="fab" onclick="ask()" title="Send">
            <svg viewBox="0 0 24 24"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.59 5.58L20 12l-8-8-8 8z"></path></svg>
          </button>
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

async function boot() {
  // Populate both dropdowns
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
  // Sync dropdowns + label
  document.getElementById('typeHero').value = value;
  document.getElementById('typeSelect').value = value;
  document.getElementById('topicText').textContent = (value==='General (All Types)') ? 'Shoulder' : value;

  // Switch to chat view
  document.getElementById('hero').style.display = 'none';
  document.getElementById('topbar').style.display = 'flex';
  document.getElementById('chatContent').style.display = 'flex';
  document.getElementById('topicPanel').style.display = 'none';

  // Fresh chat for the new type
  document.getElementById('chat').innerHTML = '';

  // Type-scoped starter pills
  renderTypePills();

  // Inform scope
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
  if(!silent) addBot('New chat started. Select a surgery type to begin.');
  await listSessions();
}

async function loadSession(id) {
  const data = await fetch('/sessions/'+id).then(r=>r.json());
  SESSION_ID = id;
  const chat = document.getElementById('chat'); chat.innerHTML='';
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
  const el = document.getElementById('chat'); el.scrollTop = el.scrollHeight;
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
    data.pills.forEach(label => {
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

@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

@app.get("/pills")
def get_pills(type: str = Query(...)):
    pills = PROCEDURE_PILLS.get(type, PROCEDURE_PILLS["General (All Types)"])
    return {"pills": pills[:4]}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".docx"):
        return JSONResponse({"ok": False, "error": "Please upload a .docx file."}, status_code=400)
    try:
        data = await file.read()
        chunks = chunk_docx(data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ingest failed: {type(e).__name__}: {e}"}, status_code=400)

    docs, ids, metas = [], [], []
    for i, ch in enumerate(chunks):
        subtype = classify_chunk(ch)
        docs.append(ch)
        ids.append(f"{file.filename}-{i}")
        metas.append({"source": file.filename, "type": subtype})
    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
    return {"ok": True, "chunks": len(docs), "file": file.filename}

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

class AskBodyModel(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None
AskBody = AskBodyModel

@app.post("/ask")
def ask(body: AskBody):
    q = (body.question or "").strip()
    if not q:
        return {"answer": "Please enter a question.", "pills": []}

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

    # Strict filter to the selected type
    where = {}
    if selected_type in PROCEDURE_KEYS and selected_type != "General (All Types)":
        where = {"type": selected_type}

    # Retrieve from Chroma
    try:
        res = collection.query(
            query_texts=[q],
            n_results=6,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
    except Exception as e:
        return {"answer": f"Search failed: {type(e).__name__}: {e}", "pills": [], "unverified": False}

    # Coverage decision (within selected type only)
    covered = likely_covered(q, docs, dists, dist_thresh=0.35)

    # Build answer
    unverified = False
    answer_text = ""

    if covered and docs:
        # STRICT: doc-only (no external info). Force model to bail if context lacks specifics.
        context = "\n\n".join(docs[:3])
        prompt = (
            "Answer ONLY using the clinic document context below. "
            "If the context is insufficient, reply with the single token: INSUFFICIENT_CONTEXT. "
            f"Scope your answer to this procedure type: {selected_type}. "
            "Write in patient-friendly language. Do NOT add external facts.\n\n"
            f"Question: {q}\n\nContext:\n{context}\n\nAnswer:"
        )
        raw = make_llm_answer(prompt, system="You are a careful medical educator.")
        if "INSUFFICIENT_CONTEXT" in raw:
            covered = False
        else:
            answer_text = summarize_to_2_or_3_sentences(raw)
            unverified = False

    if not covered:
        # Fallback: short Cleveland Clinic–style answer (explicitly unverified)
        prompt = (
            "Provide a short, accurate, patient-friendly 2–3 sentence answer about shoulder surgery, "
            "summarized in the style of Cleveland Clinic patient education content. "
            "Begin with: 'According to Cleveland Clinic,' and keep it neutral. "
            "Do NOT fabricate statistics or quotes."
            f"\n\nQuestion: {q}\n\nAnswer:"
        )
        answer_text = make_llm_answer(prompt, system="You are a careful, neutral medical explainer.")
        unverified = True

    # Adaptive follow-ups (type-scoped)
    pills = adaptive_followups(q, answer_text, selected_type)

    # Store assistant msg
    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})

    return {"answer": answer_text, "pills": pills, "unverified": unverified, "session_id": sid}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
