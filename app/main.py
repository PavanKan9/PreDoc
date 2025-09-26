from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os

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
    verified: bool  # True = grounded in uploaded docs; False = external fallback
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

# ===== Ask (concise paraphrase; verified vs. fallback; dynamic pills) =====
@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
    """
    Returns a concise 2–3 line answer (slight paraphrase) grounded in retrieved passages.
    If no relevant doc context exists, falls back to external general knowledge and marks as not verified.
    Pills adapt to the last Q+A; urgent care pill is prepended if flagged.
    """
    q = req.question.strip()
    topic = (req.topic or "shoulder").lower()

    # 1) Retrieve context (try with topic, then without)
    res = COLL.query(query_texts=[q], n_results=5, where={"topic": topic})
    docs = res.get("documents", [[]])[0]
    if not docs:
        res = COLL.query(query_texts=[q], n_results=5)
        docs = res.get("documents", [[]])[0]

    # Limit context size to keep prompts focused
    context = "\n\n---\n\n".join(docs[:3]) if docs else ""
    if len(context) > 1800:
        context = context[:1800]

    answer = ""
    verified = False

    # 2) If we have context -> summarize from doc (verified=True)
    if context:
        summary_prompt = f"""
You are a medical educator. Write a concise explanation in 2–3 lines (max ~60 words),
based only on the context below. Slightly paraphrase so it is clear and easy to read.
Do not include unrelated details, do not give medical advice, and do not mention drugs or dosages.

Context:
{context}

Question: {q}
        """.strip()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.4,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            candidate = resp.choices[0].message.content.strip()
            if len(candidate) > 420:
                candidate = candidate[:420].rstrip()
            answer = candidate
            verified = True
        except Exception:
            answer = ""

    # 3) External fallback (not verified by clinic)
    if not answer:
        ext_prompt = f"""
Patient question: {q}

Write a concise, neutral 2–3 line patient-education explanation (~60 words).
Keep it general and safe; no diagnosis, prescriptions, or dosages.
If symptoms sound urgent, suggest seeking urgent care.
Return only the explanation text.
        """.strip()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.5,
                messages=[{"role": "user", "content": ext_prompt}],
            )
            candidate = resp.choices[0].message.content.strip()
            if len(candidate) > 420:
                candidate = candidate[:420].rstrip()
            answer = (
                f"{candidate}\n\n"
                "Note: This section is general educational content that has not been verified by your clinic. "
                "Please contact your clinician for clarification."
            )
            verified = False
        except Exception:
            answer = "Sorry — I couldn’t find relevant information in the documents or generate a safe general explanation."
            verified = False

    # 4) Safety triage
    safety = triage_flags(q + "\n" + (answer or ""))

    # 5) Suggestions (model-driven with avoidance list; fallback handled in sugg.py)
    suggestions = gen_suggestions(
        q, answer, topic=topic or "shoulder", k=req.max_suggestions, avoid=req.avoid
    )

    # 6) Urgent pill on top if flagged
    urgent = {
        "urgent_care": "Seek urgent care — learn more",
        "er": "Call emergency services — what to do now",
    }
    if safety["triage"] in urgent:
        label = urgent[safety["triage"]]
        suggestions = [label] + [s for s in suggestions if s != label]
        suggestions = suggestions[: req.max_suggestions]

    return AskResp(
        answer=answer,
        practice_notes=None,  # keep field but unused
        suggestions=suggestions,
        safety=safety,
        verified=verified,
    )

# ===== Widget JS (Apple-like polish + spinner + question pills) =====
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
      --accent: #111827;
      --shadow: 0 10px 30px rgba(0,0,0,0.08);
      --radius: 14px;
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
        --accent: #e5e7eb;
        --shadow: 0 8px 28px rgba(0,0,0,0.45);
      }
    }

    body{ background: var(--bg); }
    .drqa-wrap{ max-width: 760px; margin: 0 auto; padding: 32px 20px 56px; }
    .drqa-card{ background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .drqa-head{ padding: 18px 22px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
    .drqa-title{ font-size: 18px; font-weight: 600; letter-spacing: .2px; }
    .drqa-topic{ font-size: 13px; color: var(--muted); border:1px solid var(--border); padding: 6px 10px; border-radius: 999px; background: var(--pill); }
    .drqa-body{ padding: 10px 22px 18px; }
    .drqa-messages{ display:flex; flex-direction:column; gap:12px; padding: 16px 0; min-height: 200px; }
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
    .drqa-input:focus{ border-color: #9ca3af; box-shadow: 0 0 0 3px rgba(156,163,175,0.2); }
    .drqa-btn{ padding: 11px 16px; border:0; border-radius:12px; background: var(--accent); color:#fff; cursor:pointer; font-weight:600; transition: opacity .15s ease, transform .15s ease; }
    .drqa-btn:hover{ opacity:.92; transform: translateY(-1px); }
    .drqa-btn:active{ transform: translateY(0); }

    .drqa-foot{ padding: 10px 22px 16px; color: var(--muted); font-size: 12px; }

    /* Spinner */
    .drqa-spinner { width: 22px; height: 22px; border-radius: 50%; border: 3px solid rgba(0,0,0,0.12); border-top-color: rgba(0,0,0,0.55); animation: drqa-spin 0.8s linear infinite; display: none; }
    @media (prefers-color-scheme: dark){ .drqa-spinner{ border-color: rgba(255,255,255,0.12); border-top-color: rgba(255,255,255,0.6); } }
    @keyframes drqa-spin { to { transform: rotate(360deg); } }
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
          <div id="drqa-spinner" class="drqa-spinner" aria-label="thinking"></div>
          <button type="submit" class="drqa-btn">Ask</button>
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

  var msgs = root.querySelector("#drqa-messages");
  var pills = root.querySelector("#drqa-pills");
  var form = root.querySelector("#drqa-form");
  var input = root.querySelector("#drqa-input");
  var spinner = root.querySelector("#drqa-spinner");

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
      b.textContent = label.endsWith("?") ? label : (label + "?"); // ensure questions
      b.onclick = function(){
        input.value = b.textContent;
        form.dispatchEvent(new Event("submit",{cancelable:true}));
      };
      pills.appendChild(b);
    });
  }

  function showSpinner(show){ spinner.style.display = show ? "inline-block" : "none"; }

  async function ask(q){
    addMsg(q, "user"); input.value="";
    showSpinner(true);
    try{
      const body = { question: q, topic: TOPIC, avoid: lastSuggestions };
      var res = await fetch(API + "/ask", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      var data = await res.json();
      addMsg(data.answer, "bot");
      if (data.practice_notes) addMsg(data.practice_notes, "bot");
      renderPills(data.suggestions || []);
    }catch(e){
      addMsg("Sorry — something went wrong. Please try again.", "bot");
    } finally {
      showSpinner(false);
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
    // point to your API & default topic; tweak anytime
    window.DRQA_API_URL = location.origin;
    window.DRQA_TOPIC = "shoulder";
  </script>
  <script src="/widget.js?v=8" defer></script>
</body>
</html>"""


