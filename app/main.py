# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, uuid, json

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---- Project-local utilities (safe fallback) ----
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

# Default, type-specific starter pills (first view after selecting a type)
PROCEDURE_PILLS: Dict[str, List[str]] = {
    "General (All Types)": [
        "When is shoulder arthroscopy recommended?",
        "What are the risks & complications?",
        "How long is recovery?",
        "What does PT look like after surgery?",
    ],
    "Rotator Cuff Repair": [
        "When is rotator cuff repair recommended?",
        "How long is recovery for rotator cuff repair?",
        "When can I start PT after rotator cuff repair?",
        "What are risks of rotator cuff repair?",
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
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

def summarize_to_2_or_3_sentences(text: str) -> str:
    return make_llm_answer(
        "Summarize the answer into 2–3 patient-friendly sentences without losing key specifics:\n\n" + text,
        system="You compress long answers into 2–3 clear sentences."
    )

def adaptive_followups(last_q: str, answer: str, selected_type: str) -> List[str]:
    """
    Lightweight adaptive logic:
    - Looks at the last question and the produced answer
    - Returns 4 concise, type-scoped follow-up questions
    """
    last = (last_q or "").lower()
    base = PROCEDURE_PILLS.get(selected_type or "General (All Types)", PROCEDURE_PILLS["General (All Types)"])

    # Heuristics based on common intents
    if "recover" in last or "return" in last or "heal" in last:
        return [
            f"When can I return to work/sport after {selected_type.lower()}?",
            "What milestones should I expect during rehab?",
            "How is pain typically managed during recovery?",
            "When do I transition from sling to full motion?",
        ]
    if "risk" in last or "complication" in last or "safe" in last:
        return [
            "How are risks minimized before and after surgery?",
            "What signs should prompt me to call the clinic?",
            "How common are re-injury or stiffness?",
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
            "What red flags of uncontrolled pain should I watch for?",
        ]

    # If the answer text mentions timeframes or sling → recovery themed
    if re.search(r"\b(weeks?|months?)\b", answer.lower()) or "sling" in answer.lower():
        return [
            "When do I start passive vs active motion?",
            "When can I drive again?",
            "When can I sleep without the sling?",
            "What limits should I follow at work/school?",
        ]

    # Default: type-specific starters (kept concise)
    return base[:4]

# ========= FASTAPI APP =========
app = FastAPI(title="PreDoc - Shoulder Arthroscopy Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory sessions
SESSIONS: Dict[str, Dict[str, Any]] = {}  # {id: {title, messages: [{role, content}], selected_type}}

# ========= MODELS =========
class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ========= ROUTES =========
@app.get("/", response_class=HTMLResponse)
def home():
    # Cleaner, calmer visuals. Hero appears until a type is selected.
    return HTMLResponse(f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SurgiChat · Shoulder Arthroscopy</title>
<style>
  :root {{
    --bg:#fff; --text:#0b0b0c; --muted:#6b7280; --border:#ececec;
    --accent:#0a84ff; --accent2:#ff7a18;
    --sidebar-w: 18rem;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--text);
    font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  }}
  .app {{
    display:grid; grid-template-columns: var(--sidebar-w) 1fr; height:100vh; width:100vw;
  }}
  .sidebar {{
    border-right:1px solid var(--border); padding:16px 14px; overflow:auto;
  }}
  .brand {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
  .logo {{ width:26px; height:26px; border-radius:8px; background:linear-gradient(180deg,var(--accent),var(--accent2)); }}
  .brand h1 {{ font-size:14px; font-weight:700; margin:0; letter-spacing:.2px; }}
  .new-chat {{
    width:100%; padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:#f8f8f8;
    cursor:pointer; margin-bottom:12px;
  }}
  .sideh2 {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:10px 0 6px; }}
  .chat-item {{ padding:8px 8px; border-radius:10px; cursor:pointer; }}
  .chat-item:hover {{ background:#f5f5f5; }}

  .main {{ display:flex; flex-direction:column; min-width:0; }}
  .topbar {{
    display:flex; align-items:center; justify-content:space-between; padding:14px 18px; border-bottom:1px solid var(--border);
  }}
  .title {{ font-size:16px; font-weight:700; letter-spacing:.2px; }}
  .controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}

  select, button {{
    border:1px solid var(--border); border-radius:12px; padding:9px 12px; font-size:14px; background:#fff;
  }}

  .content {{ flex:1; display:flex; flex-direction:column; overflow:hidden; }}

  /* HERO (initial state) */
  .hero {{
    flex:1; display:flex; align-items:center; justify-content:center; text-align:center; padding:40px 20px;
  }}
  .hero-inner {{ max-width:780px; }}
  .hero h2 {{
    font-size: clamp(28px, 4vw, 44px);
    line-height:1.1; margin:0 0 16px; font-weight:800; letter-spacing:-0.02em;
  }}
  .hero p {{ color:var(--muted); margin:0 0 20px; font-size:16px; }}
  .hero .selector {{
    display:flex; gap:10px; justify-content:center; align-items:center; flex-wrap:wrap;
  }}
  .hero label {{ color:#111; font-weight:600; }}
  .hero select {{ min-width:280px; }}

  /* INTRO + PILLS (after type chosen) */
  .intro {{ padding:16px 18px; border-bottom:1px solid var(--border); display:none; }}
  .intro h3 {{ margin:0 0 10px; font-size:16px; font-weight:700; }}
  .pills {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:10px; max-width:760px; }}
  @media (min-width: 1100px) {{ .pills {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }} }}
  .pill {{
    padding:10px 12px; border:1px solid var(--border); border-radius:999px; cursor:pointer; background:#fff; text-align:center; font-size:14px;
  }}
  .pill.cta {{ border-color:var(--accent2); color:var(--accent2); }}

  .chat-area {{ flex:1; overflow:auto; padding: 16px 18px; }}
  .bubble {{ max-width:820px; padding:12px 14px; border:1px solid var(--border); border-radius:14px; margin:8px 0; line-height:1.45; }}
  .bot {{ background:#fafafa; }}
  .user {{ background:#fff; margin-left:auto; border-color:#ddd; }}

  .composer {{ display:flex; gap:10px; padding: 12px 18px; border-top:1px solid var(--border); }}
  .composer input {{ flex:1; border:1px solid var(--border); border-radius:12px; padding:12px; font-size:15px; }}
  .composer button {{ background:var(--accent); color:#fff; border:none; padding: 12px 16px; border-radius:12px; cursor:pointer; }}

  .disclaimer {{ color:var(--muted); font-size:12px; margin-top:8px; }}

  .spinner {{
    width:18px; height:18px; border-radius:50%; border:3px solid #e6e6e6; border-top-color:var(--accent);
    animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-left:6px;
  }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand"><div class="logo"></div><h1>SurgiChat</h1></div>
    <button class="new-chat" onclick="newChat()">+ New chat</button>
    <div class="sideh2">Previous chats</div>
    <div id="chats"></div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="title">Patient Education · Shoulder Arthroscopy</div>
      <div class="controls" id="controlBar" style="display:none">
        <label for="type">Type</label>
        <select id="type"></select>
      </div>
    </div>

    <div class="content">
      <!-- HERO (initial prompt to select a type) -->
      <section class="hero" id="hero">
        <div class="hero-inner">
          <h2>Welcome! Select the type of surgery below:</h2>
          <p>Choose your specific shoulder arthroscopy to tailor answers and quick questions.</p>
          <div class="selector">
            <label for="typeHero">What type of Shoulder Arthroscopy</label>
            <select id="typeHero"></select>
          </div>
        </div>
      </section>

      <!-- INTRO (quick pills appear after type selection) -->
      <section class="intro" id="intro">
        <h3 id="introTitle">Quick questions</h3>
        <div class="pills" id="pills"></div>
        <div class="disclaimer">Educational use only. This does not replace medical advice.</div>
      </section>

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
  // Load types into both hero dropdown and topbar compact dropdown
  const types = await fetch('/types').then(r=>r.json()).then(d=>d.types || []);
  const selHero = document.getElementById('typeHero');
  const selTop  = document.getElementById('type');
  [selHero, selTop].forEach(sel => {{
    sel.innerHTML = '';
    types.forEach(t => {{
      const opt = document.createElement('option'); opt.value=t; opt.textContent=t; sel.appendChild(opt);
    }});
  }});
  // Hero change selects the type (initial state)
  selHero.addEventListener('change', () => handleTypeChange(selHero.value, true));
  // Topbar change adjusts selection after we're in chat state
  selTop.addEventListener('change', () => handleTypeChange(selTop.value, false));

  await listSessions();
  await newChat(true);
}}
function handleTypeChange(value, fromHero) {{
  SELECTED_TYPE = value;
  // Sync both dropdowns
  document.getElementById('typeHero').value = value;
  document.getElementById('type').value = value;

  // Transition: hide hero, show intro + control bar
  document.getElementById('hero').style.display = 'none';
  document.getElementById('intro').style.display = 'block';
  document.getElementById('controlBar').style.display = 'flex';

  // Render type-specific pills
  renderTypePills();

  // Let user know scope
  addBot('Filtering to “' + SELECTED_TYPE + '”. Ask a question or tap a pill.');
}}
function renderTypePills() {{
  const el = document.getElementById('pills');
  el.innerHTML = '';
  const title = document.getElementById('introTitle');
  title.textContent = 'Quick questions · ' + SELECTED_TYPE;

  // Ask server for defaults (keeps logic in one place if you change it later)
  fetch('/pills?type=' + encodeURIComponent(SELECTED_TYPE))
    .then(r=>r.json())
    .then(data => {{
      const pills = data.pills || [];
      pills.forEach((label, i) => {{
        const b = document.createElement('button');
        b.className = 'pill' + (i===0 ? ' cta' : '');
        b.textContent = label;
        b.onclick = () => {{
          document.getElementById('q').value = label;
          ask();
        }};
        el.appendChild(b);
      }});
    }});
}}
async function listSessions() {{
  const data = await fetch('/sessions').then(r=>r.json());
  const el = document.getElementById('chats'); el.innerHTML='';
  data.sessions.forEach(s => {{
    const d = document.createElement('div');
    d.className='chat-item';
    d.textContent = s.title || 'Untitled chat';
    d.onclick = () => loadSession(s.id);
    el.appendChild(d);
  }});
}}
async function newChat(silent=false) {{
  const data = await fetch('/sessions/new', {{method:'POST'}}).then(r=>r.json());
  SESSION_ID = data.session_id;
  if(!silent) addBot('New chat started. Select a procedure type or ask a question.');
  await listSessions();
}}
async function loadSession(id) {{
  const data = await fetch('/sessions/'+id).then(r=>r.json());
  SESSION_ID = id;
  const chat = document.getElementById('chat'); chat.innerHTML='';
  data.messages.forEach(m => {{
    if(m.role==='user') addUser(m.content); else addBot(m.content);
  }});
}}
function addUser(text) {{
  const d = document.createElement('div'); d.className='bubble user'; d.textContent=text;
  document.getElementById('chat').appendChild(d); scrollBottom();
}}
function addBot(htmlText) {{
  const d = document.createElement('div'); d.className='bubble bot'; d.innerHTML=htmlText;
  document.getElementById('chat').appendChild(d); scrollBottom();
}}
function spinner() {{
  const d = document.createElement('div'); d.className='bubble bot';
  d.innerHTML='Thinking <span class="spinner"></span>';
  document.getElementById('chat').appendChild(d); scrollBottom(); return d;
}}
function scrollBottom() {{
  const el = document.getElementById('chat'); el.scrollTop = el.scrollHeight;
}}
async function ask() {{
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  if(!SELECTED_TYPE) {{
    addBot('Please select a surgery type first.');
    return;
  }}
  addUser(q); document.getElementById('q').value='';
  const spin = spinner();
  const body = {{question: q, session_id: SESSION_ID, selected_type: SELECTED_TYPE}};
  const data = await fetch('/ask', {{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify(body)
  }}).then(r=>r.json());
  spin.remove();
  addBot(data.answer);

  // Adaptive pills (change after each answer)
  if(data.pills && data.pills.length) {{
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

  // Also refresh the intro pills to keep them type-scoped and tidy
  renderTypePills();
  await listSessions();
}}

// Boot after DOM ready
boot();
</script>
</body>
</html>
    """)

@app.get("/types")
def get_types():
    return {"types": PROCEDURE_KEYS}

@app.get("/pills")
def get_pills(type: str):
    # Return the starter pills for a given type
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
        title = v.get("title") or ("New chat")
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

# Keep legacy model for compatibility with earlier frontends
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

    # Build Chroma filter by selected type
    where = {}
    if selected_type in PROCEDURE_KEYS and selected_type != "General (All Types)":
        where = {"type": selected_type}

    # Retrieve
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

    # Coverage check (cosine distance threshold)
    covered = bool(dists and min(dists) <= 0.2)

    # Build answer
    unverified = False
    if covered and docs:
        ctx = "\n\n".join(docs[:3])
        prompt = (
            "Answer ONLY using the clinic document context below. "
            f"Keep your answer scoped to this procedure type: {selected_type}. "
            "Write in patient-friendly language. No external facts.\n\n"
            f"Question: {q}\n\nContext:\n{ctx}\n\nAnswer:"
        )
        raw = make_llm_answer(prompt, system="You are a careful medical educator.")
        answer_text = summarize_to_2_or_3_sentences(raw)
    else:
        prompt = (
            f"The clinic document did not cover this clearly for {selected_type}. "
            "Provide a short, patient-friendly answer based on general medical knowledge. "
            "Be accurate but concise.\n\n"
            f"Question: {q}\n\nAnswer:"
        )
        raw = make_llm_answer(prompt, system="You are a careful medical educator.")
        answer_text = summarize_to_2_or_3_sentences(raw)
        unverified = True

    # Special handling for recommendation questions
    if re.search(r"\bwhen\s+is\s+(a\s+)?(shoulder\s+)?(arthroscopy|repair|decompression|tenodesis|release|excision)\s+recommended\??", q.lower()):
        if covered and docs:
            ctx = "\n\n".join(docs[:3])
            prompt = (
                f"Using ONLY the context, list common indications for {selected_type} "
                "and compress to 2–3 sentences for patients. No external info.\n\n"
                f"Context:\n{ctx}\n\n2–3 sentence answer:"
            )
            answer_text = make_llm_answer(prompt, system="You are a concise medical explainer.")
            unverified = False
        else:
            prompt = (
                f"List the common indications for {selected_type} in 2–3 sentences for patients. "
                "Keep it neutral and high-level."
            )
            answer_text = make_llm_answer(prompt, system="You are a concise medical explainer.")
            unverified = True

    # Adaptive follow-ups (type-scoped, based on last q/answer)
    pills = adaptive_followups(q, answer_text, selected_type)

    # Store assistant msg
    SESSIONS[sid]["messages"].append({"role": "assistant", "content": answer_text})

    return {"answer": answer_text, "pills": pills, "unverified": unverified, "session_id": sid}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
