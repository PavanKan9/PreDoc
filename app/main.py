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
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ALLOW_ORIGINS = os.environ.get("ALLOW_ORIGINS", "*").split(",")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma")

# Retrieval strictness (lower = stricter)
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.35"))

# --- Minimal paraphrase fallback (no synonym tables) ---
def _paraphrase_once(q: str) -> str:
    """
    Single-shot neutral paraphrase to improve recall when the first retrieval is empty.
    Preserves meaning; no new claims. Returns original on failure.
    """
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Paraphrase the user's medical question in one sentence. Preserve meaning; avoid adding new claims or examples."},
                {"role": "user", "content": q},
            ],
        )
        p = (r.choices[0].message.content or "").strip()
        return p or q
    except Exception:
        return q

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

# ===== Vector store =====
persist = chromadb.PersistentClient(path=INDEX_DIR)
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
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
    # DANGER
    try:
        persist.delete_collection("shoulder_docs")
    except Exception:
        pass
    global COLL
    COLL = persist.get_or_create_collection(name="shoulder_docs", embedding_function=embedder)
    return {"ok": True}

# ===== Ingest =====
@app.post("/ingest")
async def ingest(file: UploadFile = File(...), topic: str = Form("shoulder")):
    """
    Upload a .docx and ingest into the collection.
    """
    if not file.filename.lower().endswith(".docx"):
        return {"ok": False, "error": "Only .docx is supported for now."}

    path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())

    chunks = read_docx_chunks(path)
    if not chunks:
        return {"ok": False, "error": "No text found in document."}

    ids = [f"{file.filename}:{i}" for i in range(len(chunks))]
    metas = [{"title": file.filename, "topic": topic} for _ in chunks]
    COLL.upsert(ids=ids, documents=chunks, metadatas=metas)
    return {"ok": True, "count": len(chunks)}

# ===== Ask =====
@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    q = (req.question or "").strip()
    topic = (req.topic or "shoulder").lower()
    max_k = req.max_suggestions if isinstance(req.max_suggestions, int) else 4

    NO_MATCH_MESSAGE = (
        "I couldn’t find this answered in the clinic’s provided materials. "
        "You can try rephrasing your question, or ask your clinician directly."
    )

    # 1) Retrieval (global for single-doc setup)
    try:
        res = COLL.query(query_texts=[q], n_results=5, include=["documents","metadatas","distances"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] or []
    except Exception:
        docs = []
        metas = []
        dists = []
    # If nothing retrieved (or empty strings), paraphrase once and retry
    if (not docs) or all((not (d or "").strip()) for d in docs):
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            res = COLL.query(query_texts=[q2], n_results=5, include=["documents","metadatas","distances"])
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] or []

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

    # Verified via embedding distance (top hit under threshold)
    verified = False
    try:
        # consider retrieved distance if available
        if 'dists' in locals() and dists:
            verified = (float(dists[0]) is not None) and (float(dists[0]) < DISTANCE_THRESHOLD)
    except Exception:
        verified = (answer != NO_MATCH_MESSAGE)

    # Preserve original answer for suggestions to avoid skew from a prefix
    answer_for_suggestions = answer

    # If not verified and the answer isn't already the standard not-found message, prefix explicitly
    if (not verified) and (answer.strip() != NO_MATCH_MESSAGE.strip()):
        answer = "**Not found in your uploaded clinic material.** Here is general information:\n\n" + answer

    # 4) Safety triage
    try:
        safety = triage_flags(q + "\n" + answer) or {"triage": None}
    except Exception:
        safety = {"triage": None}

    # 5) Suggestions (generate, then filter shoulder-only)
    try:
        suggestions = gen_suggestions(
            q, answer_for_suggestions, topic=topic, k=max_k, avoid=req.avoid
        ) or []
    except Exception:
        suggestions = []

    SHOULDER_DEFAULTS = [
        "What is shoulder arthroscopy?",
        "When is it recommended?",
        "What are the risks?",
        "How long is recovery?",
    ]
    if not suggestions:
        suggestions = SHOULDER_DEFAULTS[:max_k]

    return AskResp(
        answer=answer,
        practice_notes=None,
        suggestions=suggestions[:max_k],
        safety=safety,
        verified=verified,
    )

# ===== Peek endpoint (dev helper) =====
@app.get("/peek")
def peek(q: str, topic: str = "shoulder"):
    try:
        scoped = COLL.query(query_texts=[q], n_results=3, where={"topic": topic})
        global_q = COLL.query(query_texts=[q], n_results=3)
        return {
            "scoped_docs": scoped.get("documents", [[]])[0],
            "global_docs": global_q.get("documents", [[]])[0],
        }
    except Exception as e:
        return {"error": str(e)}

# ===== Minimal widget =====
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(BASIC_PAGE)

@app.get("/widget.js", response_class=PlainTextResponse)
def widget_js(v: Optional[str] = None):
    return PlainTextResponse(WIDGET_JS, media_type="application/javascript")

# ===== Static assets (very small embeddable) =====

WIDGET_JS = r"""
(() => {
  const API = (window.DRQA_API_URL || location.origin).replace(/\/+$/,'');
  const TOPIC = window.DRQA_TOPIC || 'shoulder';

  const root = document.getElementById('drqa-root');
  if (!root) return;

  root.innerHTML = `
    <style>
      .drqa{font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111; max-width:680px; margin:24px auto; background:#fff; border:1px solid #e2e2e2; border-radius:12px; overflow:hidden; box-shadow:0 10px 20px rgba(0,0,0,.06)}
      .drqa-h{padding:14px 16px; background:#f8f8fa; border-bottom:1px solid #eee; font-weight:600}
      .drqa-b{padding:16px}
      .drqa-q{display:flex; gap:8px}
      .drqa-q input{flex:1; padding:10px 12px; border:1px solid #ddd; border-radius:8px}
      .drqa-q button{padding:10px 14px; border:0; background:#0b5fff; color:#fff; border-radius:8px; cursor:pointer}
      .drqa-a{white-space:pre-wrap; padding:12px 0}
      .drqa-s{display:flex; gap:8px; flex-wrap:wrap; margin-top:8px}
      .drqa-chip{padding:6px 10px; border:1px solid #ddd; border-radius:999px; cursor:pointer; background:#fafafa}
      .drqa-disclaimer{margin-top:8px; color:#555; font-size:12px}
    </style>
    <div class="drqa">
      <div class="drqa-h">PreDoc — Patient Education</div>
      <div class="drqa-b">
        <div class="drqa-q">
          <input id="drqa-in" placeholder="Ask about shoulder arthroscopy…" />
          <button id="drqa-go">Ask</button>
        </div>
        <div id="drqa-a" class="drqa-a"></div>
        <div id="drqa-s" class="drqa-s"></div>
        <div id="drqa-dis" class="drqa-disclaimer"></div>
      </div>
    </div>
  `;

  const $in = root.querySelector('#drqa-in');
  const $go = root.querySelector('#drqa-go');
  const $a = root.querySelector('#drqa-a');
  const $s = root.querySelector('#drqa-s');
  const $dis = root.querySelector('#drqa-dis');

  async function run(q){
    $a.textContent = '…';
    $s.innerHTML = '';
    $dis.textContent = '';

    try{
      const r = await fetch(API + '/ask', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({question:q, topic: TOPIC})
      });
      const j = await r.json();
      $a.textContent = j.answer || '';
      $dis.textContent = j.disclaimer || '';
      (j.suggestions||[]).forEach(ch => {
        const b = document.createElement('button');
        b.className = 'drqa-chip';
        b.textContent = ch;
        b.onclick = () => run(ch);
        $s.appendChild(b);
      });
    }catch(e){
      $a.textContent = 'Error. Please try again.';
    }
  }

  $go.onclick = () => run($in.value.trim());
  $in.onkeydown = (e) => { if(e.key === 'Enter'){ run($in.value.trim()); } };
})();
"""

BASIC_PAGE = r"""<!doctype html>
<html>
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
  <script src="/widget.js?v=17" defer></script>
</body>
</html>"""
