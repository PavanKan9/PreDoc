from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os, re

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

from .safety import triage_flags
from .sugg import gen_suggestions
from .parsing import read_docx_chunks

# ===== Env & constants =====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is required"

ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "").split(",") if o.strip()] or ["*"]
DATA_DIR = os.environ.get("DATA_DIR", "/data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma")

# Retrieval strictness (lower = stricter)
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.35"))

# ===== App & middleware =====
client = OpenAI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Vector DB (Chroma persistent) =====
persist = chromadb.PersistentClient(path=INDEX_DIR)
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
COLL = persist.get_or_create_collection(name="shoulder_docs", embedding_function=embedder)

# ===== Pydantic models =====
class AskReq(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    topic: Optional[str] = "shoulder"
    max_suggestions: int = 4
    avoid: List[str] = []  # chips to avoid repeating

class AskResp(BaseModel):
    answer: str
    practice_notes: Optional[str] = None
    suggestions: List[str]
    safety: dict
    verified: bool  # True = grounded in uploaded docs; False = not covered
    disclaimer: str = "Educational information only — not medical advice."

# ===== Utility / debug =====
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats():
    try:
        count = COLL.count()
    except Exception as e:
        count = f"error: {e}"
    return {"collection": "shoulder_docs", "count": count}

@app.post("/reset")
def reset():
    # DANGER: wipes the shoulder_docs collection
    try:
        persist.delete_collection("shoulder_docs")
    except Exception:
        pass
    global COLL
    COLL = persist.get_or_create_collection(name="shoulder_docs", embedding_function=embedder)
    return {"ok": True}

# ===== Ingest =====
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), topic: str = Form("shoulder")):
    """
    Upload a Word doc; extract chunks; add to Chroma.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    chunks = read_docx_chunks(path)
    ids = [f"{topic}-{os.path.basename(path)}-{i}" for i in range(len(chunks))]

    # Add in batches to avoid payload limits
    B = 64
    for i in range(0, len(chunks), B):
        COLL.add(
            documents=chunks[i:i+B],
            ids=ids[i:i+B],
            metadatas=[{"topic": topic}] * len(chunks[i:i+B]),
        )
    return {"added": len(chunks)}

# ===== Ask (concise paraphrase when grounded; otherwise “not covered”) =====
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician for guidance."
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def clamp_to_3_sentences(text: str) -> str:
    """Ensure the final answer is 2–3 short sentences max."""
    text = text.strip()
    if not text:
        return text
    sents = _SENT_SPLIT.split(text)
    # remove empty fragments
    sents = [s.strip() for s in sents if s.strip()]
    # keep max 3 sentences
    sents = sents[:3]
    # if model gave just 1 long sentence, that's fine; otherwise join
    return " ".join(sents)

def join_top_docs(doc_lists: List[str], max_chars: int = 1800) -> str:
    """Join top doc chunks into a single snippet with a soft char cap."""
    snippet = "\n\n---\n\n".join(doc_lists[:3])
    return snippet[:max_chars]

# ---------- Helpers (single-doc setup: global retrieval) ----------

def _normalize(txt: str) -> str:
    """Trim, collapse whitespace, and strip obvious Q:/A: headers."""
    import re
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def filter_suggestions_in_doc(suggestions: List[str], limit: int = 4) -> List[str]:
    """Keep only pills that the (global) retriever can actually answer."""
    good: List[str] = []
    for s in (suggestions or []):
        try:
            res = COLL.query(query_texts=[s], n_results=1)  # GLOBAL (no where)
            docs = res.get("documents", [[]])[0]
            if docs:
                good.append(s)
                if len(good) >= limit:
                    break
        except Exception:
            pass
    return good

CURATED_DEFAULTS = [
    "What is shoulder arthroscopy?",
    "When is it recommended?",
    "What are the risks?",
    "How long is recovery?",
]

NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician directly."
)

# ----------------------------- /ask -----------------------------
@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
    q = (req.question or "").strip()
    topic = (req.topic or "shoulder").lower()
    max_k = req.max_suggestions if isinstance(req.max_suggestions, int) else 4

    NO_MATCH_MESSAGE = (
        "I couldn’t find this answered in the clinic’s provided materials. "
        "You can try rephrasing your question, or ask your clinician directly."
    )

    # 1) Retrieval (global for single-doc setup)
    try:
        res = COLL.query(query_texts=[q], n_results=5)
        docs = res.get("documents", [[]])[0]
    except Exception:
        docs = []

    if not docs:
        return AskResp(
            answer=NO_MATCH_MESSAGE,
            practice_notes=None,
            suggestions=[
                "What is shoulder arthroscopy?",
                "When is it recommended?",
                "What are the risks?",
                "How long is recovery?",
            ][:max_k],
            safety={"triage": None},
            verified=False,
        )

    # 2) Clean and join top chunks
    def _normalize(txt: str) -> str:
        import re
        if not isinstance(txt, str):
            return ""
        t = txt.strip()
        t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    clean_docs = [_normalize(d) for d in docs[:3] if isinstance(d, str) and d.strip()]
    context = "\n\n---\n\n".join(clean_docs)
    if not context:
        return AskResp(
            answer=NO_MATCH_MESSAGE,
            practice_notes=None,
            suggestions=[
                "What is shoulder arthroscopy?",
                "When is it recommended?",
                "What are the risks?",
                "How long is recovery?",
            ][:max_k],
            safety={"triage": None},
            verified=False,
        )
    if len(context) > 1800:
        context = context[:1800]

    # 3) Summarize strictly from clinic material
    summary_messages = [
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
        {
            "role": "user",
            "content": f"Question: {q}\n\nMaterial:\n{context}",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=summary_messages,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception:
        answer = NO_MATCH_MESSAGE

    verified = (answer != NO_MATCH_MESSAGE)

    # 4) Safety triage
    try:
        safety = triage_flags(q + "\n" + answer) or {"triage": None}
    except Exception:
        safety = {"triage": None}

    # 5) Suggestions (generate, then filter shoulder-only)
    try:
        suggestions = gen_suggestions(
            q, answer, topic=topic, k=max_k, avoid=req.avoid
        ) or []
    except Exception:
        suggestions = []

    SHOULDER_DEFAULTS = [
        "What is shoulder arthroscopy?",
        "When is it recommended?",
        "What are the risks?",
        "How long is recovery?",
    ]
    OFF_TOPIC_TERMS = {"knee", "hip", "spine", "ankle", "wrist", "elbow", "back", "neck"}

    def _shoulder_only(sugs, limit):
        out = []
        for s in (sugs or []):
            low = (s or "").lower()
            if any(t in low for t in OFF_TOPIC_TERMS):
                continue
            label = s.strip()
            if label and not label.endswith("?"):
                label += "?"
            out.append(label)
            if len(out) >= limit:
                break
        return out

    filtered = _shoulder_only(suggestions, max_k)
    if not filtered:
        filtered = SHOULDER_DEFAULTS[:max_k]
    suggestions = filtered

    return AskResp(
        answer=answer,
        practice_notes=None,
        suggestions=suggestions[:max_k],
        safety=safety,
        verified=verified,
    )


# ----------------------------- /peek -----------------------------

@app.get("/peek")
def peek(q: str, topic: str = "shoulder"):
    try:
        scoped = COLL.query(query_texts=[q], n_results=3, where={"topic": topic})
        global_q = COLL.query(query_texts=[q], n_results=3)
        return {
            "scoped_docs": scoped.get("documents", [[]])[0],
            "scoped_metas": scoped.get("metadatas", [[]])[0],
            "global_docs": global_q.get("documents", [[]])[0],
            "global_metas": global_q.get("metadatas", [[]])[0],
        }
    except Exception as e:
        return {"error": str(e)}


# ===== Widget JS (unchanged from your version) =====
@app.get("/widget.js", response_class=PlainTextResponse)
def widget_js():
    return """
(function(){
  var API = (window.DRQA_API_URL || (location.origin));
  var ROOT_ID = (window.DRQA_ROOT_ID || "drqa-root");
  var TOPIC = (window.DRQA_TOPIC || "shoulder");

  var root = document.getElementById(ROOT_ID);
  if(!root){ root = document.createElement("div"); root.id = ROOT_ID; document.body.appendChild(root); }

  root.innerHTML = `
  <style>
    :root {
      --bg: #f5f5f7;
      --card: #ffffff;
      --text: #111111;
      --muted: #6b7280;
      --border: #e5e7eb;
      --pill: #f9fafb;
      --pill-border: #e5e7eb;
      --user: #e8eefc;
      --bot: #f6f7f8;
      --accent: #ff9900; /* Amazon orange */
      --shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0b0b0c;
        --card: #111113;
        --text: #f5f5f7;
        --muted: #9aa1aa;
        --border: #1f2125;
        --pill: #0f1012;
        --pill-border: #24262b;
        --user: #12233f;
        --bot: #151617;
        --shadow: 0 8px 28px rgba(0,0,0,0.45);
      }
    }

    body { background: var(--bg); }

    body, .drqa-card, .drqa-bubble, .drqa-pill, .drqa-input, .drqa-title, .drqa-send {
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .drqa-wrap{ max-width: 760px; margin: 0 auto; padding: 32px 20px 56px; }
    .drqa-card{ background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 14px; box-shadow: var(--shadow); overflow: hidden; }
    .drqa-head{ padding: 18px 22px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
    .drqa-title{ font-size: 18px; font-weight: 600; letter-spacing: .2px; }
    .drqa-topic{ font-size: 13px; color: var(--muted); border:1px solid var(--border); padding: 6px 10px; border-radius: 999px; background: var(--pill); }
    .drqa-body{ padding: 10px 22px 18px; }
    .drqa-messages{ display:flex; flex-direction:column; gap:12px; padding: 16px 0; min-height: 200px; max-height: 58vh; overflow-y: auto; -webkit-overflow-scrolling: touch; }
    .drqa-bubble{ max-width: 85%; padding: 11px 13px; line-height: 1.45; border-radius: 14px; white-space: pre-wrap; border: 1px solid var(--border); opacity: 0; transform: translateY(4px); animation: drqa-in .18s ease forwards; }
    .drqa-bubble.user{ align-self:flex-end; background: var(--user); }
    .drqa-bubble.bot{ align-self:flex-start; background: var(--bot); }
    @keyframes drqa-in { to { opacity:1; transform: translateY(0); } }

    .drqa-pills{ display:flex; flex-wrap:wrap; gap:8px; padding: 6px 0 12px; }
    .drqa-pill{ border:1px solid var(--pill-border); background: var(--pill); border-radius: 999px; padding: 8px 12px; cursor:pointer; font-size: 14px; transition: all .15s ease; user-select:none; }
    .drqa-pill:hover{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .drqa-pill:active{ transform: translateY(0); }

    .drqa-form{ display:flex; gap:10px; align-items:center; padding-top: 8px; border-top: 1px solid var(--border); margin-top: 8px; }
    .drqa-input{ flex:1; padding: 12px 14px; border:1px solid var(--border); border-radius: 12px; background: transparent; color: var(--text); outline: none; transition: border-color .15s ease, box-shadow .15s ease; font-size: 15px; }

    .drqa-send{
      width: 40px; height: 40px;
      display: inline-flex; align-items: center; justify-content: center;
      border: 0; border-radius: 9999px;
      background: var(--accent); color: #111;
      cursor: pointer; font-weight: 700;
      transition: transform .15s ease, opacity .15s ease;
      position: relative; flex: 0 0 auto;
    }
    .drqa-send:hover{ transform: translateY(-1px); }
    .drqa-send:active{ transform: translateY(0); }
    .drqa-send:disabled{ opacity: .6; cursor: not-allowed; }

    .drqa-send__icon{ font-size: 18px; line-height: 1; transform: translateY(-1px); display: block; }
    .drqa-send__spinner{
      position: absolute; inset: 0; margin: auto; width: 20px; height: 20px;
      border-radius: 50%; border: 3px solid rgba(0,0,0,0.15); border-top-color: rgba(0,0,0,0.55);
      animation: drqa-spin .8s linear infinite; display: none;
    }
    @keyframes drqa-spin { to { transform: rotate(360deg); } }

    .drqa-send.is-loading .drqa-send__icon{ display: none; }
    .drqa-send.is-loading .drqa-send__spinner{ display: block; }

    .drqa-foot{ padding: 10px 22px 16px; color: var(--muted); font-size: 12px; }
  </style>

  <div class="drqa-wrap">
    <div class="drqa-card">
      <div class="drqa-head">
        <div class="drqa-title">Patient Education</div>
        <div class="drqa-topic">Topic: <span id="drqa-topic-text"></span></div>
      </div>

      <div class="drqa-body">
        <div id="drqa-messages" class="drqa-messages"></div>
        <div id="drqa-pills" class="drqa-pills"></div>

        <form id="drqa-form" class="drqa-form">
          <input id="drqa-input" class="drqa-input" type="text" placeholder="Ask about your shoulder…" autocomplete="off">
          <button id="drqa-send" class="drqa-send" type="submit" aria-label="Send">
            <span class="drqa-send__icon">↑</span>
            <span class="drqa-send__spinner" aria-hidden="true"></span>
          </button>
        </form>
      </div>

      <div class="drqa-foot">
        Educational information only — not medical advice.
      </div>
    </div>
  </div>
  `;

  var topicEl = root.querySelector("#drqa-topic-text");
  topicEl.textContent = (TOPIC.charAt(0).toUpperCase() + TOPIC.slice(1));

  var msgs   = root.querySelector("#drqa-messages");
  var pills  = root.querySelector("#drqa-pills");
  var form   = root.querySelector("#drqa-form");
  var input  = root.querySelector("#drqa-input");
  var send   = root.querySelector("#drqa-send");

  function addMsg(text, who){
    var d = document.createElement("div");
    d.className = "drqa-bubble " + (who==="user" ? "user" : "bot");
    d.textContent = text;
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
  }

  var lastSuggestions = [];
  function renderPills(arr){
    lastSuggestions = Array.isArray(arr) ? arr.slice(0) : [];
    pills.innerHTML = "";
    (arr||[]).forEach(function(label){
      var b = document.createElement("button");
      b.type = "button";
      b.className = "drqa-pill";
      b.textContent = label.endsWith("?") ? label : (label + "?");
      b.onclick = function(){ input.value = b.textContent; form.dispatchEvent(new Event("submit",{cancelable:true})); };
      pills.appendChild(b);
    });
  }

  function setLoading(loading){
    if(loading){
      send.classList.add("is-loading");
      send.setAttribute("aria-busy","true");
      send.disabled = true;
      input.disabled = true;
    }else{
      send.classList.remove("is-loading");
      send.removeAttribute("aria-busy");
      send.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  async function ask(q){
    addMsg(q, "user"); input.value="";
    setLoading(true);
    try{
      const body = { question: q, topic: TOPIC, avoid: lastSuggestions };
      var res = await fetch(API + "/ask", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      var data = await res.json();
      addMsg(data.answer || "No answer available.", "bot");
      renderPills(data.suggestions || []);
    }catch(e){
      addMsg("Sorry — something went wrong. Please try again.", "bot");
    } finally {
      setLoading(false);
    }
  }

  // Defaults (first render)
  renderPills([
    "What is shoulder arthroscopy?",
    "When is it recommended?",
    "What are the risks?",
    "How long is recovery?"
  ]);

  form.addEventListener("submit", function(ev){
    ev.preventDefault();
    var q = input.value.trim();
    if(q) ask(q);
  });

})();
""".strip()

# ===== Minimal home (let the widget own the UI) =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PreDoc — Patient Education Chat</title>
  <meta name="color-scheme" content="light dark">
  <style>body{margin:0;background:#f5f5f7}</style>
</head>
<body>
  <div id="drqa-root"></div>
  <script>
    window.DRQA_API_URL = location.origin;
    window.DRQA_TOPIC = "shoulder";
  </script>
  <script src="/widget.js?v=16" defer></script>
</body>
</html>"""
