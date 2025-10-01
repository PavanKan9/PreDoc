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

# ---- Safety fallback ----
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

# ========= PROCEDURE TYPES =========
PROCEDURE_TYPES = {
    "General (All Types)": [],
    "Rotator Cuff Repair": ["rotator cuff", "supraspinatus", "infraspinatus", "subscapularis", "teres minor", "rc tear"],
    "SLAP Repair": ["slap tear", "superior labrum", "biceps anchor", "type ii slap", "labral superior", "slap repair"],
    "Bankart (Anterior Labrum) Repair": ["bankart", "anterior labrum", "anterior instability", "glenoid labrum anterior"],
    "Posterior Labrum Repair": ["posterior labrum", "posterior instability", "reverse bankart"],
    "Biceps Tenodesis/Tenotomy": ["biceps tenodesis", "tenotomy", "biceps tendon", "lhb", "biceps tendinopathy"],
    "Subacromial Decompression (SAD)": ["subacromial decompression", "acromioplasty", "impingement", "s.a.d"],
    "Distal Clavicle Excision": ["distal clavicle excision", "dce", "mumford", "ac joint resection"],
    "Capsular Release": ["capsular release", "adhesive capsulitis", "frozen shoulder", "arthroscopic release"],
    "Debridement/Diagnostic Only": ["debridement", "diagnostic arthroscopy", "synovectomy"],
}
PROCEDURE_KEYS = list(PROCEDURE_TYPES.keys())

# ========= HELPERS =========
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

def chunk_docx(file_bytes: bytes) -> List[str]:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    chunks, size, overlap = [], 1000, 150
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def make_llm_answer(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a careful medical explainer that ONLY uses the provided context."},
            {"role":"user","content":prompt}
        ],
        temperature=0,
        max_tokens=320,
    )
    return resp.choices[0].message.content.strip()

def summarize_doc_answer(question: str, context: str) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"Material:\n{context}\n\n"
        "Answer in 2–3 clear sentences (45–80 words) using ONLY this material. "
        "Do not add anything not present in the material."
    )
    return make_llm_answer(prompt)

# ========= FASTAPI APP =========
app = FastAPI(title="Patient Education")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= SESSIONS =========
SESSIONS: Dict[str, Dict[str, Any]] = {}

class AskBody(BaseModel):
    question: str
    session_id: Optional[str] = None
    selected_type: Optional[str] = None

# ========= ROUTES =========
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Patient Education</title>
<style>
  .hero h1 {
    font-size: clamp(34px, 4.5vw, 50px);
    font-weight: 800;
    color: #000; /* all black */
    margin-bottom: 12px;
  }
  .hero select {
    min-width:280px; border:2px solid #ff7a18; border-radius:12px; padding:10px 12px;
    color:inherit; background:#fff;
  }
</style>
</head>
<body>
  <section class="hero">
    <div class="hero-inner">
      <h1>Welcome! Select the type of surgery below:</h1>
      <div class="selector">
        <select id="typeHero"></select>
      </div>
    </div>
  </section>
  <script>
    async function boot() {
      const types = await fetch('/types').then(r=>r.json()).then(d=>d.types||[]);
      const selHero = document.getElementById('typeHero');
      types.forEach(t=>{
        const o=document.createElement('option'); o.value=t; o.textContent=t; selHero.appendChild(o);
      });
    }
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
    data = await file.read()
    chunks = chunk_docx(data)
    docs, ids, metas = [], [], []
    for i, ch in enumerate(chunks):
        subtype = classify_chunk(ch)
        docs.append(ch)
        ids.append(f"{file.filename}-{i}")
        metas.append({"source": file.filename, "type": subtype})
    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)
    return {"ok": True, "chunks": len(docs)}

@app.post("/ask")
def ask(body: AskBody):
    q = (body.question or "").strip()
    sid = body.session_id or uuid.uuid4().hex[:10]
    if sid not in SESSIONS:
        SESSIONS[sid] = {"messages": []}
    selected_type = body.selected_type or "General (All Types)"

    where = {}
    if selected_type in PROCEDURE_KEYS and selected_type != "General (All Types)":
        where = {"type": selected_type}

    res = collection.query(
        query_texts=[q],
        n_results=5,
        where=where if where else None,
        include=["documents"]
    )
    docs = (res.get("documents") or [[]])[0]

    if not docs:
        return {"answer": "I couldn’t find this answered in the clinic’s provided materials.", "unverified": True}

    context = "\n\n---\n\n".join(docs[:3])
    answer = summarize_doc_answer(q, context)

    return {"answer": answer, "unverified": False, "session_id": sid}

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"
