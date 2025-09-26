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
    verified: bool
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
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    chunks = read_docx_chunks(path)
    ids = [f"{topic}-{os.path.basename(path)}-{i}" for i in range(len(chunks))]

    B = 64
    for i in range(0, len(chunks), B):
        COLL.add(
            documents=chunks[i:i+B],
            ids=ids[i:i+B],
            metadatas=[{"topic": topic}] * len(chunks[i:i+B]),
        )
    return {"added": len(chunks)}

# ===== Answering helpers =====
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician for guidance."
)

def _normalize(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _paraphrase_once(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Paraphrase the user's medical question in one sentence. Preserve meaning; don't add new claims."},
                {"role": "user", "content": q},
            ],
        )
        p = (r.choices[0].message.content or "").strip()
        return p or q
    except Exception:
        return q

def _retrieve(q: str, n: int = 5, topic: Optional[str] = None):
    try:
        kwargs = {"query_texts": [q], "n_results": n}
        if topic:
            kwargs["where"] = {"topic": topic}
        res = COLL.query(**kwargs)
        return res.get("documents", [[]])[0]
    except Exception:
        return []

def _build_context(docs: List[str], max_chars: int = 1800) -> str:
    clean_docs = [_normalize(d) for d in docs[:3] if isinstance(d, str) and d.strip()]
    context = "\n\n---\n\n".join(clean_docs)
    return context[:max_chars] if context else ""

def _summarize_from_context(q: str, context: str) -> str:
    if not context:
        return NO_MATCH_MESSAGE
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
        {"role": "user", "content": f"Question: {q}\n\nMaterial:\n{context}"},
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=summary_messages,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return NO_MATCH_MESSAGE

# Body part detection
_BODY_PARTS = {
    "shoulder": {"shoulder"},
    "knee": {"knee"},
    "hip": {"hip"},
    "elbow": {"elbow"},
    "wrist": {"wrist"},
    "ankle": {"ankle"},
    "spine": {"spine", "back"},
    "neck": {"neck", "cervical"},
    "hand": {"hand", "hands"},
    "foot": {"foot", "feet"},
}

def _mentioned_parts(text: str) -> set:
    low = (text or "").lower()
    found = set()
    for part, tokens in _BODY_PARTS.items():
        if any(t in low for t in tokens):
            found.add(part)
    return found

# ----------------------------- /ask -----------------------------
@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
    q = (req.question or "").strip()
    topic = (req.topic or "shoulder").lower()
    max_k = req.max_suggestions if isinstance(req.max_suggestions, int) else 4

    NO_MATCH_MESSAGE_LOCAL = NO_MATCH_MESSAGE

    # 1) Retrieval
    docs = _retrieve(q, n=5, topic=topic)

    if (not docs) or all((not (d or "").strip()) for d in docs):
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            docs = _retrieve(q2, n=5, topic=topic)

    if not docs:
        return AskResp(
            answer=NO_MATCH_MESSAGE_LOCAL,
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

    # 2) Context
    context = _build_context(docs, max_chars=1800)
    if not context:
        return AskResp(
            answer=NO_MATCH_MESSAGE_LOCAL,
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

    # 2b) Cross-topic guard
    parts_in_q = _mentioned_parts(q)
    if parts_in_q and (topic not in parts_in_q):
        ctx_low = context.lower()
        if not any(any(tok in ctx_low for tok in _BODY_PARTS[p]) for p in parts_in_q):
            return AskResp(
                answer=NO_MATCH_MESSAGE_LOCAL,
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

    # 3) Summarize
    answer = _summarize_from_context(q, context)

    if answer.strip() == NO_MATCH_MESSAGE_LOCAL.strip():
        q2 = _paraphrase_once(q)
        if q2 and q2 != q:
            docs2 = _retrieve(q2, n=5, topic=topic)
            ctx2 = _build_context(docs2, max_chars=1800) if docs2 else ""
            if ctx2:
                answer2 = _summarize_from_context(q2, ctx2)
                if answer2.strip():
                    answer = answer2

    verified = (answer.strip() != NO_MATCH_MESSAGE_LOCAL.strip())

    # 4) Safety
    try:
        safety = triage_flags(q + "\n" + answer) or {"triage": None}
    except Exception:
        safety = {"triage": None}

    # 5) Suggestions
    try:
        suggestions = gen_suggestions(q, answer, topic=topic, k=max_k, avoid=req.avoid) or []
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

# ===== Widget JS (unchanged) =====
@app.get("/widget.js", response_class=PlainTextResponse)
def widget_js():
    return """(function(){ /* ... UI code unchanged ... */ })()""".strip()

# ===== Minimal home =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><html><body><div id="drqa-root"></div><script src="/widget.js?v=16" defer></script></body></html>"""
