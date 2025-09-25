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

# ===== Widget JS (spinner + preserved line breaks + dynamic pills) =====
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
    /* Simple spinner (Chrome-like) */
    .drqa-spinner {
      width: 22px; height: 22px; border-radius: 50%;
      border: 3px solid rgba(0,0,0,0.1);
      border-top-color: rgba(0,0,0,0.55);
      animation: drqa-spin 0.8s linear infinite;
      display: none; margin-left: 8px;
    }
    @keyframes drqa-spin { to { transform: rotate(360deg); } }
  </style>
  <div class="drqa-box" style="max-width:680px;margin:0 auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif">
    <div id="drqa-messages" style="display:flex;flex-direction:column;gap:12px;padding:8px 0;"></div>
    <div id="drqa-pills" style="display:flex;flex-wrap:wrap;gap:8px;padding:6px 0 12px;"></div>
    <form id="drqa-form" style="display:flex;gap:8px;align-items:center;">
      <input id="drqa-input" type="text" placeholder="Ask about your shoulder…" autocomplete="off" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:8px;">
      <div id="drqa-spinner" class="drqa-spinner" aria-label="thinking"></div>
      <button type="submit" style="padding:10px 14px;border:0;border-radius:8px;background:#111827;color:#fff;cursor:pointer;">Ask</button>
    </form>
    <div style="margin-top:8px;font-size:12px;color:#6b7280;">Educational information only — not medical advice.</div>
  </div>`;

  var msgs = root.querySelector("#drqa-messages");
  var pills = root.querySelector("#drqa-pills");
  var form = root.querySelector("#drqa-form");
  var input = root.querySelector("#drqa-input");
  var spinner = root.querySelector("#drqa-spinner");

  function addMsg(text, who){
    var d = document.createElement("div");
    d.style.padding="10px 12px"; d.style.borderRadius="12px"; d.style.lineHeight="1.35";
    d.style.whiteSpace = "pre-wrap"; // preserve newlines
    if(who==="user"){ d.style.background="#eef2ff"; d.style.alignSelf="flex-end"; }
    else { d.style.background="#f4f4f5"; }
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
      b.textContent = label.endsWith("?") ? label : (label + "?"); // force question style
      b.style.border="1px solid #ddd"; b.style.borderRadius="999px"; b.style.padding="6px 10px";
      b.style.cursor="pointer"; b.style.fontSize="14px"; b.style.background="#fff";
      b.onclick = function(){ input.value = b.textContent; form.dispatchEvent(new Event("submit",{cancelable:true})); };
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

      if (data.verified === false) {
        addMsg("Note: This answer is general educational content and hasn’t been verified by your clinic.", "bot");
      }

      renderPills(data.suggestions || []);
    }catch(e){
      addMsg("Sorry — something went wrong. Please try again.", "bot");
    } finally {
      showSpinner(false);
    }
  }

  // Defaults before first answer
  renderPills(["What is shoulder arthroscopy?","When is it recommended?","What are the risks?","How long is recovery?"]);

  form.addEventListener("submit", function(ev){
    ev.preventDefault();
    var q = input.value.trim();
    if(q) ask(q);
  });
})();
""".strip()

@app.get("/", response_model=None, response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en"><head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Patient Education Chat</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f9fafb;display:flex;justify-content:center;padding:40px;}
  </style>
</head>
<body>
  <div id="drqa-root"></div>
  <script src="/widget.js?v=7" defer></script>
</body>
</html>"""
