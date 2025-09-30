# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, uuid, json

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---- Project-local utilities (keep as-is in your repo) ----
try:
    from .safety import triage_flags
except Exception:
    # Fallback no-op if running stand-alone
    def triage_flags(text: str) -> Dict[str, Any]:
        return {"blocked": False, "reasons": []}

# ========= ENV & GLOBALS =========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")

os.makedirs(DATA_DIR, exist_ok=True)
# Ensure chroma subdir exists on first boot so persistence doesn’t error
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

# ========= SHOULDER ARTHROSCOPY SUBTYPES & TAGGING =========
PROCEDURE_TYPES = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff", "supraspinatus", "infraspinatus", "subscapularis", "teres minor", "rc tear"],
    "SLAP Repair": ["slap tear", "superior labrum", "biceps anchor", "type ii slap", "labral superior"],
    "Bankart (Anterior Labrum) Repair": ["bankart", "anterior labrum", "anterior instability", "glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum", "posterior instability", "reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis", "tenotomy", "biceps tendon", "lhb"],
    "Subacromial Decompression (SAD)": ["subacromial decompression", "acromioplasty", "impingement", "s.a.d"],
    "Distal Clavicle Excision": ["distal clavicle excision", "dce", "mumford", "ac joint resection"],
    "Capsular Release": ["capsular release", "adhesive capsulitis", "frozen shoulder", "arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement", "diagnostic arthroscopy", "synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

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
    # Minimal, fast chunker for .docx text
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError("python-docx is required. pip install python-docx")

    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    # Chunk by ~1000 chars with overlap
    chunks, size, overlap = [], 1000, 150
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap
    return chunks

def make_llm_answer(prompt: str, system: str = "You are a concise medical explainer.") -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

def summarize_to_2_or_3_sentences(text: str) -> str:
    prompt = (
        "Summarize the following answer into 2–3 sentences, patient-friendly, "
        "without losing key specifics:\n\n" + text
    )
    return make_llm_answer(prompt, system="You compress long answers into 2–3 clear sentences.")

def pills_for_intro() -> List[str]:
    # Ensure first pill is the requested one.
    return [
        "When is it recommended?",
        "What are the risks & complications?",
        "What does recovery look like?",
        "What is the PT plan after surgery?",
    ]

def followup_pills_from_answer(answer: str) -> List[str]:
    # Keep pills as questions only (no content).
    base = [
        "Am I a good candidate?",
        "How long before I can drive/work out?",
        "What pain control is typical?",
        "When do stitches come out?",
    ]
    return base[:4]

# ========= FASTAPI =========
app = FastAPI(title="PreDoc - Shoulder Arthroscopy Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions for "Previous Chats" (persist lightly if desired)
SESSIONS: Dict[str, Dict[str, Any]] = {}  # {session_id: {"title": str, "messages": [..]}}

# ========= MODELS =========
class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None  # must be one of PROCEDURE_KEYS or None

# ========= ROUTES =========

@app.get("/", response_class=HTMLResponse)
def home():
    # Full-screen UI with left sidebar (prev chats) and right chat. Apple-like clean.
    # Includes subtype dropdown and 2x2 pill layout.
    return HTMLResponse(content=f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SurgiChat · Shoulder Arthroscopy</title>
<style>
  :root {{
    --bg: #ffffff;
    --text: #0b0b0c;
    --muted: #6b7280;
    --accent: #0a84ff;       /* Apple-ish blue */
    --accent-2: #ff7a18;     /* Orange for CTA/pills */
    --border: #e5e7eb;
    --sidebar-w: 20vw;       /* ~1/6 */
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji;
  }}
  .app {{
    display: grid; grid-template-columns: var(--sidebar-w) 1fr; height: 100vh; width: 100vw;
  }}
  .sidebar {{
    border-right: 1px solid var(--border); padding: 20px; overflow-y: auto;
  }}
  .brand {{
    display:flex; align-items:center; gap:10px; margin-bottom: 18px;
  }}
  .logo {{
    width: 28px; height: 28px; border-radius: 8px; background: linear-gradient(180deg,var(--accent), var(--accent-2));
  }}
  .brand h1 {{ font-size: 16px; font-weight: 700; margin:0; }}
  .new-chat {{
    display:block; width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:12px;
    background:#f9fafb; text-align:center; cursor:pointer; margin-bottom:12px;
  }}
  .chat-list h2 {{
    font-size: 12px; text-transform: uppercase; letter-spacing:.08em; color: var(--muted); margin: 14px 0 8px;
  }}
  .chat-item {{
    padding:10px 8px; border-radius:10px; cursor:pointer;
  }}
  .chat-item:hover {{ background:#f3f4f6; }}
  .main {{
    display:flex; flex-direction:column; height:100%; width:100%;
  }}
  .topbar {{
    display:flex; align-items:center; justify-content:space-between;
    padding: 18px 22px; border-bottom: 1px solid var(--border);
  }}
  .title {{ font-size: 18px; font-weight:700; letter-spacing:.2px; }}
  .controls {{ display:flex; gap:12px; align-items:center; flex-wrap: wrap; }}
  select, button {{
    border:1px solid var(--border); border-radius:12px; padding:10px 12px; font-size:14px; background:#fff;
  }}
  .content {{
    flex:1; display:flex; flex-direction:column; overflow: hidden;
  }}
  .intro {{
    padding: 18px 22px; border-bottom: 1px solid var(--border);
  }}
  .pills {{
    display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:10px; max-width:720px;
  }}
  .pill {{
    padding:10px 12px; border:1px solid var(--border); border-radius:999px; cursor:pointer;
    background: #fff; text-align:center; font-size:14px;
  }}
  .pill.cta {{ border-color: var(--accent-2); color: var(--accent-2); }}
  @media (min-width: 1100px) {{
    .pills {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
  }}
  .chat-area {{ flex:1; overflow-y:auto; padding: 18px 22px; }}
  .bubble {{
    max-width: 800px; padding:12px 14px; border:1px solid var(--border); border-radius:14px; margin:8px 0;
    line-height:1.45;
  }}
  .bot {{ background:#f9fafb; }}
  .user {{ background:#ffffff; margin-left: auto; border-color:#d1d5db; }}
  .composer {{
    display:flex; gap:10px; padding: 14px 22px; border-top: 1px solid var(--border);
  }}
  .composer input {{
    flex:1; border:1px solid var(--border); border-radius:12px; padding:12px; font-size:15px;
  }}
  .composer button {{
    background: var(--accent); color:white; border:none; padding: 12px 16px; border-radius:12px; cursor:pointer;
  }}
  .disclaimer {{
    color: var(--muted); font-size: 12px; margin-top:8px;
  }}
  .spinner {{
    width:18px; height:18px; border-radius:50%; border:3px solid #e5e7eb; border-top-color: var(--accent);
    animation: spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:6px;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="logo"></div>
      <h1>SurgiChat</h1>
    </div>
    <button class="new-chat" onclick="newChat()">+ New chat</button>
    <div class="chat-list">
      <h2>Previous chats</h2>
      <div id="chats"></div>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="title">Patient Education · Shoulder Arthroscopy</div>
      <div class="controls">
        <label for="type">What type of Shoulder Arthroscopy</label>
        <select id="type"></select>
      </div>
    </div>

    <div class="content">
      <div class="intro" id="intro">
        <div style="font-weight:600; margin-bottom:8px;">Quick questions</div>
        <div class="pills" id="pills"></div>
        <div class="disclaimer">Educational use only. This does not replace medical advice.</div>
      </div>
      <div class="chat-area" id="chat"></div>
      <div class="composer">
        <input id="q" placeholder="Ask about your procedure…" onkeydown="if(event.key==='Enter') ask()"/>
        <button onclick="ask()">Send</button>
      </div>
    </div>
  </main>
</div>

<script>
let SESSION_ID = null;
let SELECTED_TYPE = null;

async function boot() {{
  await loadTypes();
  await listSessions();
  newChat(true);
  renderIntroPills();
}}
async function loadTypes() {{
  const res = await fetch('/types');
  const data = await res.json();
  const sel = document.getElementById('type');
  sel.innerHTML = '';
  data.types.forEach(t => {{
    const opt = document.createElement('option');
    opt.value = t;
    opt.textContent = t;
    sel.appendChild(opt);
  }});
  sel.addEventListener('change', () => {{
    SELECTED_TYPE = sel.value;
    // FIXED: avoid Python f-string seeing {{}} — no template literal with ${...}
    addBot('Filtering to “' + SELECTED_TYPE + '”. Ask a question or tap a pill.');
  }});
  SELECTED_TYPE = sel.value;
}}
async function listSessions() {{
  const res = await fetch('/sessions');
  const data = await res.json();
  const el = document.getElementById('chats');
  el.innerHTML = '';
  data.sessions.forEach(s => {{
    const d = document.createElement('div');
    d.className = 'chat-item';
    d.textContent = s.title || 'Untitled chat';
    d.onclick = () => loadSession(s.id);
    el.appendChild(d);
  }});
}}
async function newChat(silent=false) {{
  const res = await fetch('/sessions/new', {{method:'POST'}});
  const data = await res.json();
  SESSION_ID = data.session_id;
  if(!silent) addBot('New chat started. Select a procedure type or ask a question.');
  await listSessions();
}}
async function loadSession(id) {{
  const res = await fetch('/sessions/' + id);
  const data = await res.json();
  SESSION_ID = id;
  const chat = document.getElementById('chat');
  chat.innerHTML = '';
  data.messages.forEach(m => {{
    if(m.role==='user') addUser(m.content); else addBot(m.content);
  }});
}}
function addUser(text) {{
  const d = document.createElement('div'); d.className = 'bubble user'; d.textContent = text;
  document.getElementById('chat').appendChild(d);
  scrollBottom();
}}
function addBot(htmlText) {{
  const d = document.createElement('div'); d.className = 'bubble bot'; d.innerHTML = htmlText;
  document.getElementById('chat').appendChild(d);
  scrollBottom();
}}
function spinner() {{
  const d = document.createElement('div'); d.className='bubble bot'; d.innerHTML='Thinking <span class="spinner"></span>';
  document.getElementById('chat').appendChild(d);
  scrollBottom();
  return d;
}}
function scrollBottom() {{
  const el = document.getElementById('chat');
  el.scrollTop = el.scrollHeight;
}}
function renderIntroPills() {{
  const el = document.getElementById('pills');
  el.innerHTML = '';
  const pills = {json.dumps(pills_for_intro())};
  pills.forEach((label, i) => {{
    const b = document.createElement('button');
    b.className = 'pill' + (i===0 ? ' cta' : '');
    b.textContent = label;
    b.onclick = () => {{
      document.getElementById('q').value = (i===0 ? 'When is shoulder arthroscopy recommended?' : label);
      ask();
    }};
    el.appendChild(b);
  }});
}}
async function ask() {{
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  addUser(q);
  document.getElementById('q').value='';
  const spin = spinner();
  const body = {{question: q, session_id: SESSION_ID, selected_type: SELECTED_TYPE}};
  const res = await fetch('/ask', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
  const data = await res.json();
  spin.remove();
  addBot(data.answer);
  if(data.pills && data.pills.length) {{
    // show 2x2 pills under the bot message
    const wrap = document.createElement('div'); wrap.style.marginTop='6px';
    const grid = document.createElement('div'); grid.className='pills';
    data.pills.forEach(label => {{
      const b = document.createElement('button'); b.className='pill'; b.textContent=label;
      b.onclick = () => {{ document.getElementById('q').value = label; ask(); }};
      grid.appendChild(b);
    }});
    wrap.appendChild(grid);
    document.getElementById('chat').appendChild(wrap);
  }}
  if(data.unverified) {{
    addBot('<div class="disclaimer">This information is not verified by the clinic; please contact your provider with questions.</div>');
  }}
  await listSessions();
}}
boot();
</script>
</body>
</html>
    """)

@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

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
    # Return id + title only
    out = [{"id": k, "title": v.get("title") or (v["messages"][0]["content"][:40] + "…"
            if v.get("messages") else "New chat")} for k, v in SESSIONS.items()]
    # newest first
    out.sort(key=lambda x: x["id"], reverse=True)
    return {"sessions": out}

@app.post("/sessions/new")
def new_session():
    sid = uuid.uuid4().hex[:10]
    SESSIONS[sid] = {"title": "New chat", "messages": []}
    return {"session_id": sid}

@app.get("/sessions/{sid}")
def read_session(sid: str):
    sess = SESSIONS.get(sid, {"messages": []})
    return {"messages": sess["messages"]}

@app.post("/ask")
def ask(body: AskBody):
    q = (body.question or "").strip()
    if not q:
        return {"answer": "Please enter a question.", "pills": []}

    safety = triage_flags(q)
    if safety.get("blocked"):
        return {"answer": "I can’t help with that request.", "pills": [], "unverified": False}

    # Store session/user msg
    sid = body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"title": "New chat", "messages": []}
    SESSIONS[sid]["messages"].append({"role": "user", "content": q})
    if not SESSIONS[sid].get("title") or SESSIONS[sid]["title"] == "New chat":
        SESSIONS[sid]["title"] = q[:60]

    # Build Chroma filter by selected type
    where = {}
    selected = body.selected_type
    if selected and selected in PROCEDURE_KEYS and selected != "General (All Types)":
        where = {"type": selected}

    # Retrieve
    try:
        res = collection.query(
            query_texts=[q],
            n_results=6,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
    except Exception as e:
        return {"answer": f"Search failed: {type(e).__name__}: {e}", "pills": [], "unverified": False}

    # Heuristic coverage check:
    # Consider "covered by clinic" if we have at least one chunk within a similarity distance threshold
    covered = False
    threshold = 0.2  # cosine distance; tweak as needed
    if dists:
        best = min(dists)
        covered = best <= threshold

    answer_text = ""
    unverified = False

    if covered and docs:
        # Compose a grounded answer from top 3 chunks
        context = "\n\n".join(docs[:3])
        prompt = (
            "You are answering based ONLY on the provided clinic document context.\n"
            "Write a clear, patient-friendly answer to the user question.\n"
            "If the question is 'When is shoulder arthroscopy recommended?', explain typical indications.\n"
            "Do NOT add external facts.\n\n"
            f"Question: {q}\n\nContext:\n{context}\n\nAnswer:"
        )
        raw = make_llm_answer(prompt, system="You are a careful medical educator.")
        answer_text = summarize_to_2_or_3_sentences(raw)
        unverified = False
    else:
        # External fallback (unverified) ONLY if nowhere in doc
        prompt = (
            "The clinic document did not cover this. Provide a short, patient-friendly answer "
            "about shoulder arthroscopy based on general medical knowledge. Be accurate but concise."
            f"\n\nQuestion: {q}\n\nAnswer:"
        )
        raw = make_llm_answer(prompt, system="You are a careful medical educator.")
        answer_text = summarize_to_2_or_3_sentences(raw)
        unverified = True

    # Special handling for the intro pill (force a crisp, indications-focused answer)
    if re.search(r"\bwhen\s+is\s+(a\s+)?shoulder\s+arthroscopy\s+recommended\??", q.lower()):
        # Prefer doc, but if not covered, provide general indications and mark as appropriate.
        if covered and docs:
            ctx = "\n\n".join(docs[:3])
            prompt = (
                "Using ONLY the context, list the common indications for shoulder arthroscopy "
                "(rotator cuff tears, labral tears/instability, impingement, biceps tendon pathology, etc.). "
                "Then compress to 2–3 sentences for patients. No external info.\n\n"
                f"Context:\n{ctx}\n\n2–3 sentence answer:"
            )
            answer_text = make_llm_answer(prompt, system="You are a concise medical explainer.")
            unverified = False
        else:
            prompt = (
                "List the common indications for shoulder arthroscopy in 2–3 sentences for patients. "
                "Keep it neutral and high-level."
            )
            answer_text = make_llm_answer(prompt, system="You are a concise medical explainer.")
            unverified = True

    # Build pills (questions only)
    pills = followup_pills_from_answer(answer_text)

    html_answer = answer_text
    if unverified:
        html_answer += ""

    # Store assistant msg
    SESSIONS[sid]["messages"].append({"role": "assistant", "content": html_answer})

    return {"answer": html_answer, "pills": pills, "unverified": unverified, "session_id": sid}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
