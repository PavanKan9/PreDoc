from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Set
import os, re, string

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

UNVERIFIED_SUFFIX = (
    "This information is not verified by the clinic; please contact your provider with questions."
)

# Internal sentinel used only for logic (never shown to users)
NO_MATCH_MESSAGE = (
    "I couldn’t find this answered in the clinic’s provided materials. "
    "You can try rephrasing your question, or ask your clinician directly."
)

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
    verified: bool  # True = grounded in uploaded docs; False = external/unverified
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

# ===== Helpers =====
def _normalize(txt: str) -> str:
    """Trim, collapse whitespace, and strip obvious Q:/A: headers."""
    if not isinstance(txt, str):
        return ""
    t = txt.strip()
    t = re.sub(r"(?:^|\n)(Q:|Question:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:^|\n)(A:|Answer:).*?$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _paraphrase_once(q: str) -> str:
    """One-shot neutral paraphrase used only as a fallback for retrieval."""
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

_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","without","by","about","from",
    "is","are","was","were","be","being","been","do","does","did","can","could","should","would",
    "how","what","when","why","where","which","who","whom","that","this","these","those","it",
    "as","at","into","over","under","than","then","so","if","but"
}

def _keywordify(q: str, limit: int = 10) -> str:
    """Extract simple keyword string to improve retrieval."""
    low = (q or "").lower()
    trans = str.maketrans("", "", string.punctuation)
    tokens = [t.translate(trans) for t in low.split()]
    keywords = [t for t in tokens if t and (t not in _STOPWORDS)]
    # keep order and dedupe
    seen = set()
    out = []
    for t in keywords:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return " ".join(out) if out else (q or "")

def _retrieve_many(queries: List[str], n_each: int = 6, topic: Optional[str] = None) -> List[str]:
    """Run multiple queries and merge unique docs preserving rank."""
    ids_seen: Set[int] = set()
    merged: List[str] = []
    for q in queries:
        if not q:
            continue
        try:
            kwargs = {"query_texts": [q], "n_results": n_each}
            if topic:
                kwargs["where"] = {"topic": topic}
            res = COLL.query(**kwargs)
            docs = res.get("documents", [[]])[0]
        except Exception:
            docs = []
        for d in docs:
            if isinstance(d, str) and d.strip():
                # Use object id to dedupe by content identity (fallback to hash)
                key = id(d)
                if key in ids_seen:
                    continue
                ids_seen.add(key)
                merged.append(d)
    return merged

def _build_context(docs: List[str], max_docs: int = 6, max_chars: int = 3200) -> str:
    clean_docs = [_normalize(d) for d in docs[:max_docs] if isinstance(d, str) and d.strip()]
    context = "\n\n---\n\n".join(clean_docs)
    return context[:max_chars] if context else ""

def _summarize_from_context(q: str, context: str) -> str:
    """Summarize strictly from provided clinic material; otherwise return NO_MATCH_MESSAGE."""
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
            temperature=0.1,
            messages=summary_messages,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return NO_MATCH_MESSAGE

def _external_answer(q: str) -> str:
    """
    External fallback: produce a neutral, evidence-based 2–3 sentence answer
    without referencing clinic docs. The caller appends the unverified suffix.
    """
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content":
                 "You are a medical explainer. Answer in 2–3 clear sentences (45–80 words). "
                 "Be neutral and educational, avoid clinic-specific guidance, and do not include disclaimers."},
                {"role": "user", "content": q},
            ],
        )
        ans = (r.choices[0].message.content or "").strip()
        return ans or "Here is general educational information based on typical medical guidance."
    except Exception:
        return "Here is general educational information based on typical medical guidance."

# Body part detection for cross-topic guard (removed generic 'back' to avoid false positives)
_BODY_PARTS = {
    "shoulder": {"shoulder", "rotator cuff", "labrum", "biceps tendon"},
    "knee": {"knee"},
    "hip": {"hip"},
    "elbow": {"elbow"},
    "wrist": {"wrist"},
    "ankle": {"ankle"},
    "spine": {"spine"},
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

    # ---- Hybrid retrieval (robust to paraphrase) ----
    q_par = _paraphrase_once(q)
    q_kw  = _keywordify(q)
    queries = [q]
    if q_par and q_par != q:
        queries.append(q_par)
    if q_kw and q_kw not in queries:
        queries.append(q_kw)

    docs = _retrieve_many(queries, n_each=8, topic=topic)

    # If still nothing, answer externally (UNVERIFIED)
    if not docs:
        ext = _external_answer(q)
        answer = f"{ext}\n\n{UNVERIFIED_SUFFIX}"
        verified = False
        safety = {"triage": None}
        try:
            safety = triage_flags(q + "\n" + answer) or {"triage": None}
        except Exception:
            pass
        suggestions = [
            "What is shoulder arthroscopy?",
            "When is it recommended?",
            "What are the risks?",
            "How long is recovery?",
        ][:max_k]
        return AskResp(answer=answer, practice_notes=None, suggestions=suggestions, safety=safety, verified=verified)

    # Build context
    context = _build_context(docs, max_docs=6, max_chars=3200)
    if not context:
        ext = _external_answer(q)
        answer = f"{ext}\n\n{UNVERIFIED_SUFFIX}"
        verified = False
        safety = {"triage": None}
        try:
            safety = triage_flags(q + "\n" + answer) or {"triage": None}
        except Exception:
            pass
        suggestions = [
            "What is shoulder arthroscopy?",
            "When is it recommended?",
            "What are the risks?",
            "How long is recovery?",
        ][:max_k]
        return AskResp(answer=answer, practice_notes=None, suggestions=suggestions, safety=safety, verified=verified)

    # Cross-topic guard (now less aggressive)
    parts_in_q = _mentioned_parts(q)
    if parts_in_q and (topic not in parts_in_q):
        ctx_low = context.lower()
        if not any(any(tok in ctx_low for tok in _BODY_PARTS[p]) for p in parts_in_q):
            ext = _external_answer(q)
            answer = f"{ext}\n\n{UNVERIFIED_SUFFIX}"
            verified = False
            safety = {"triage": None}
            try:
                safety = triage_flags(q + "\n" + answer) or {"triage": None}
            except Exception:
                pass
            suggestions = [
                "What is shoulder arthroscopy?",
                "When is it recommended?",
                "What are the risks?",
                "How long is recovery?",
            ][:max_k]
            return AskResp(answer=answer, practice_notes=None, suggestions=suggestions, safety=safety, verified=verified)

    # Summarize strictly from context
    answer = _summarize_from_context(q, context)

    # If the model says "not covered", retry with paraphrase context; otherwise verified
    verified = True
    if answer.strip() == NO_MATCH_MESSAGE.strip():
        # Try paraphrase-only retrieval context as a last doc-grounded attempt
        docs2 = _retrieve_many([q_par, q_kw], n_each=8, topic=topic)
        ctx2 = _build_context(docs2, max_docs=6, max_chars=3200) if docs2 else ""
        if ctx2:
            answer2 = _summarize_from_context(q, ctx2)
            if answer2.strip() != NO_MATCH_MESSAGE.strip():
                answer = answer2
            else:
                verified = False
        else:
            verified = False

    # If still not verified, external fallback
    if not verified:
        ext = _external_answer(q)
        answer = f"{ext}\n\n{UNVERIFIED_SUFFIX}"

    # Safety triage
    try:
        safety = triage_flags(q + "\n" + answer) or {"triage": None}
    except Exception:
        safety = {"triage": None}

    # Suggestions (shoulder-only filter)
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
    OFF_TOPIC_TERMS = {"knee", "hip", "spine", "ankle", "wrist", "elbow", "neck"}  # removed "back"

    def _shoulder_only(sugs, limit):
        out = []
        for s in (sugs or []):
            low = (s or "").lower()
            if any(t in low for t in OFF_TOPIC_TERMS):
                continue
            label = (s or "").strip()
            if label and not label.endswith("?"):
                label += "?"
            if label:
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
        sco
