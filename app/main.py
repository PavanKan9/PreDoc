from fastapi import FastAPI, UploadFile, File, Query
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, io, re, uuid, sys

# ---- OpenAI ----
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
        metadata={"hnsw:space": "cosine"},
    )
except Exception:
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---- Optional modules ----
try:
    from .safety import triage_flags
except Exception:
    def triage_flags(text: str): return {"blocked": False, "reasons": []}

try:
    from .sugg import gen_suggestions
except Exception:
    def gen_suggestions(q, a, topic=None, k=4, avoid=None): return []

try:
    from .parsing import read_docx_chunks
    HAVE_READ_DOCX = True
except Exception:
    HAVE_READ_DOCX = False

def chunk_docx_bytes(b: bytes) -> List[str]:
    import docx
    doc = docx.Document(io.BytesIO(b))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    out, size, overlap, i = [], 1000, 150, 0
    while i < len(text):
        out.append(text[i:i+size]); i += size - overlap
    return out

ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]

# ========= PROCEDURE TYPES & PILLS =========
PROCEDURE_TYPES = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff","rcr","supraspinatus","infraspinatus","subscapularis","teres minor"],
    "SLAP Repair": ["slap","superior labrum","biceps anchor","type ii slap"],
    "Bankart (Anterior Labrum) Repair": ["bankart","anterior labrum","anterior instability"],
    "Posterior Labrum Repair": ["posterior labrum","posterior instability","reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis","tenotomy","biceps tendon","lhb"],
    "Subacromial Decompression (SAD)": ["subacromial decompression","sad","acromioplasty","impingement"],
    "Distal Clavicle Excision": ["distal clavicle excision","dce","mumford","ac joint resection"],
    "Capsular Release": ["capsular release","adhesive capsulitis","frozen shoulder"],
    "Debridement/Diagnostic Only": ["debridement","diagnostic arthroscopy","synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

PROCEDURE_PILLS = {
    "General (All Types)": ["What is shoulder arthroscopy?","When is it recommended?","What are the risks?"],
    "Rotator Cuff Repair": ["When is rotator cuff repair recommended?","How long is recovery for rotator cuff repair?","What are early rehab precautions after cuff repair?"],
    "SLAP Repair": ["When is SLAP repair recommended?","How long is recovery for SLAP repair?","What are early rehab precautions after SLAP repair?"],
    "Bankart (Anterior Labrum) Repair": ["When is Bankart repair recommended?","How long is recovery for Bankart repair?","What instability precautions after Bankart repair?"],
    "Posterior Labrum Repair": ["When is posterior labrum repair recommended?","How long is recovery for posterior labrum repair?","What motions should I avoid early after posterior repair?"],
    "Biceps Tenodesis/Tenotomy": ["When is biceps tenodesis recommended?","How long is recovery for biceps tenodesis?","What are lifting precautions after tenodesis?"],
    "Subacromial Decompression (SAD)": ["When is subacromial decompression recommended?","How long is recovery for SAD?","When can I return to work after SAD?"],
    "Distal Clavicle Excision": ["When is distal clavicle excision recommended?","How long is recovery for DCE?","When can I bench press after DCE?"],
    "Capsular Release": ["When is capsular release recommended?","How long is recovery for capsular release?","What is the PT plan after capsular release?"],
    "Debridement/Diagnostic Only": ["When is arthroscopic debridement recommended?","How long is recovery after debridement?","What can I do the first two weeks after debridement?"],
}

# ========= RED-FLAG DETECTION =========
RED_FLAG_MESSAGE = (
    "This may be a post-operative red flag. "
    "<strong>Please contact the clinic directly:</strong> "
    "SF office — (415) 353-6400 &nbsp;•&nbsp; Marin office — (415) 886-8538. "
    "If you feel unsafe, call emergency services."
)

_re_temp = re.compile(r"\b(temp|fever)\b.*?(100\.4|101|102|103)", re.I)
_re_redness = re.compile(r"\b(red|warm|inflamed|hot)\b.*\b(worse|spreading|tracking|increasing)\b", re.I)
_re_streaks = re.compile(r"red(\s+)?streak", re.I)
_re_drain = re.compile(r"\b(drain|ooze|discharge|fluid|liquid|gunk|leak|weeping)\b", re.I)
_re_foul = re.compile(r"\b(bad|foul|weird|strong|unusual)\s*(smell|odor)\b", re.I)
_re_pus = re.compile(r"\b(pus|purulent|yellow|green|thick)\s*(fluid|drain|discharge)?\b", re.I)
_re_open = re.compile(r"(open|split|gaping|separate|dehis)\w*\s*(incision|wound|site)?", re.I)
_re_pain = re.compile(r"\b(worse|increasing|more)\s+(pain|sore|ache)\b", re.I)

def _detect_red_flags(t: str) -> bool:
    if not t: return False
    t = t.lower()
    wound_words = any(w in t for w in ["incision","wound","cut","site","scar","surgery","operation"])
    has_symptom = any([
        _re_temp.search(t), _re_redness.search(t), _re_streaks.search(t),
        _re_drain.search(t), _re_foul.search(t), _re_pus.search(t),
        _re_open.search(t), _re_pain.search(t)
    ])
    return bool((has_symptom and wound_words) or _re_temp.search(t))

# ========= HELPER FUNCTIONS =========
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician directly."
)
STOPWORDS = {"a","the","and","or","to","of","for","in","on","with","after","before","is","are","be"}
def _normalize(t): return re.sub(r"\s+"," ",t.strip()) if isinstance(t,str) else ""
def _retrieve(q, n=8, topic="shoulder", tsel=None):
    try:
        where={"topic":topic}
        if tsel and tsel!="General (All Types)": where["type"]=tsel
        res=collection.query(query_texts=[q],n_results=n,where=where)
        return res.get("documents",[[]])[0]
    except Exception: return []
def _build_context(docs,maxc=1800):
    out=[_normalize(d) for d in docs if d and str(d).strip()]
    return "\n\n---\n\n".join(out)[:maxc]
def _summarize(q,ctx):
    if not client: return NO_MATCH_MESSAGE
    if not ctx: return NO_MATCH_MESSAGE
    try:
        r=client.chat.completions.create(
            model="gpt-4o-mini",temperature=0.15,
            messages=[
                {"role":"system","content":"You are a concise orthopedic explainer; use only provided text. 3–5 sentences, no pleasantries."},
                {"role":"user","content":f"Question: {q}\n\nMaterial:\n{ctx}"}
            ])
        return (r.choices[0].message.content or "").strip()
    except Exception: return NO_MATCH_MESSAGE

# ========= FASTAPI APP =========
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

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

@app.post("/ask")
def ask(body: AskBody):
    q_raw=(body.question or "").strip()
    if not q_raw:
        return {"answer":"Please enter a question.","pills":[],"unverified":False}
    # Red-flag stop
    if _detect_red_flags(q_raw):
        sid=body.session_id or uuid.uuid4().hex[:10]
        if sid not in SESSIONS: SESSIONS[sid]={"title":"","messages":[]}
        SESSIONS[sid]["messages"].append({"role":"user","content":q_raw})
        SESSIONS[sid]["messages"].append({"role":"assistant","content":RED_FLAG_MESSAGE})
        return {"answer":RED_FLAG_MESSAGE,"pills":[],"unverified":False,"session_id":sid}

    sid=body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS: SESSIONS[sid]={"title":"","messages":[]}
    SESSIONS[sid]["messages"].append({"role":"user","content":q_raw})
    SESSIONS[sid]["title"]=SESSIONS[sid].get("title") or q_raw[:60]
    selected=body.selected_type or "General (All Types)"
    docs=_retrieve(q_raw,10,"shoulder",selected)
    if not docs:
        return {"answer":NO_MATCH_MESSAGE,"pills":[],"unverified":True,"session_id":sid}
    ctx=_build_context(docs)
    ans=_summarize(q_raw,ctx)
    SESSIONS[sid]["messages"].append({"role":"assistant","content":ans})
    return {"answer":ans,"pills":PROCEDURE_PILLS.get(selected,[])[:3],"unverified":False,"session_id":sid}

@app.get("/")
def home():
    return HTMLResponse("""<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Patient Education</title>
<style>
body{margin:0;font-family:"Inter",system-ui,-apple-system,"Segoe UI",Roboto;}
.app{display:grid;grid-template-columns:15rem 1fr;height:100vh;}
.sidebar{background:#f7f7f8;border-right:1px solid #eaeaea;padding:16px;overflow:auto;}
.new-chat{width:100%;padding:10px;margin-bottom:14px;border:1px solid #ccc;border-radius:12px;background:#fff;cursor:pointer;font-weight:600;}
.main{display:flex;flex-direction:column;}
.chat{flex:1;padding:20px;overflow:auto;}
.bubble{padding:12px;margin:6px 0;border-radius:12px;max-width:70%;}
.user{background:#fff;border:1px solid #ccc;margin-left:auto;}
.bot{background:#f5f5f5;border:1px solid #ddd;}
.pills{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:10px;}
.composer{display:flex;border-top:1px solid #ccc;padding:10px;}
.composer input{flex:1;border:none;outline:none;font-size:16px;}
.fab{border:none;background:#ff7a18;color:#fff;border-radius:50%;width:42px;height:42px;cursor:pointer;}
</style></head><body>
<div class="app">
<aside class="sidebar">
<button class="new-chat" onclick="newChat()">+ New chat</button>
<div id="chats"></div></aside>
<main class="main">
<div class="chat" id="chat"></div>
<div class="composer"><div class="pills" id="pills"></div>
<input id="q" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')ask()"/>
<button class="fab" onclick="ask()">➤</button></div>
</main></div>
<script>
let SESSION_ID=null;
async function listSessions(){const d=await fetch('/sessions').then(r=>r.json());
const e=document.getElementById('chats');e.innerHTML='';
d.sessions.forEach(s=>{const div=document.createElement('div');div.style.cursor='pointer';
div.textContent=s.title;div.onclick=()=>loadSession(s.id);e.appendChild(div);});}
async function newChat(){const d=await fetch('/sessions/new',{method:'POST'}).then(r=>r.json());
SESSION_ID=d.session_id;document.getElementById('chat').innerHTML='';await listSessions();}
async function loadSession(id){const d=await fetch('/sessions/'+id).then(r=>r.json());
SESSION_ID=id;const c=document.getElementById('chat');c.innerHTML='';
d.messages.forEach(m=>{const b=document.createElement('div');b.className='bubble '+(m.role==='user'?'user':'bot');b.innerHTML=m.content;c.appendChild(b);});}
function addUser(t){const b=document.createElement('div');b.className='bubble user';b.textContent=t;document.getElementById('chat').appendChild(b);}
function addBot(t){const b=document.createElement('div');b.className='bubble bot';b.innerHTML=t;document.getElementById('chat').appendChild(b);}
async function ask(){const q=document.getElementById('q').value.trim();if(!q)return;
addUser(q);document.getElementById('q').value='';const body={question:q,session_id:SESSION_ID};
const d=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
addBot(d.answer);}
listSessions();
</script></body></html>""")
