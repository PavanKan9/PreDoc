from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

from .safety import triage_flags
from .sugg import gen_suggestions
from .parsing import read_docx_chunks

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is required"

ALLOW_ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "").split(",") if o.strip()] or ["*"]
DATA_DIR = os.environ.get("DATA_DIR", "/data")
INDEX_DIR = os.path.join(DATA_DIR, "chroma")

client = OpenAI()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

persist = chromadb.PersistentClient(path=INDEX_DIR)
embedder = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
COLL = persist.get_or_create_collection(name="shoulder_docs", embedding_function=embedder)

class AskReq(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    topic: Optional[str] = "shoulder"
    max_suggestions: int = 4

class AskResp(BaseModel):
    answer: str
    practice_notes: Optional[str] = None
    suggestions: List[str]
    safety: dict
    disclaimer: str = "Educational information only — not medical advice."

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), topic: str = Form("shoulder")):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    chunks = read_docx_chunks(path)
    ids = []
    for i, c in enumerate(chunks):
        ids.append(f"{topic}-{os.path.basename(path)}-{i}")
    B = 64
    for i in range(0, len(chunks), B):
        COLL.add(
            documents=chunks[i:i+B],
            ids=ids[i:i+B],
            metadatas=[{"topic": topic}] * len(chunks[i:i+B]),
        )
    return {"added": len(chunks)}

@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
    q = req.question.strip()
    res = COLL.query(query_texts=[q], n_results=5, where={"topic": req.topic})
    top_docs = res.get("documents", [[]])[0]
    context = "\n\n".join(top_docs[:3]) if top_docs else ""

    system = (
        "You are a careful patient-education assistant. "
        "Answer using ONLY the provided context. "
        "No diagnosis or treatment plans. No drug names or dosages. "
        "Keep it plain-language and short. "
        "If relevant, add a 'Practice notes' block with ≤3 gentle tips. "
        "End with: 'Educational only — not medical advice.'"
    )
    user = f"Patient question: {q}\n\nContext:\n{context}\n\nWrite a concise answer (2–4 sentences)."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    full = resp.choices[0].message.content.strip()

    answer, practice = full, None
    if "\nPractice notes" in full:
        parts = full.split("\nPractice notes", 1)
        answer = parts[0].strip()
        practice = "Practice notes" + parts[1]

    safety = triage_flags(q + "\n" + (answer or ""))
    suggestions = gen_suggestions(q, answer, topic=req.topic or "shoulder", k=req.max_suggestions)

    urgent = {"urgent_care": "Seek urgent care — learn more", "er": "Call emergency services — what to do now"}
    if safety["triage"] in urgent:
        label = urgent[safety["triage"]]
        suggestions = [label] + [s for s in suggestions if s != label]
        suggestions = suggestions[: req.max_suggestions]

    return AskResp(answer=answer, practice_notes=practice, suggestions=suggestions, safety=safety)

@app.get("/widget.js", response_class=PlainTextResponse)
def widget_js():
    return """(function(){
      var API = (window.DRQA_API_URL || (location.origin));
      var ROOT_ID = (window.DRQA_ROOT_ID || "drqa-root");
      var TOPIC = (window.DRQA_TOPIC || "shoulder");
      var root = document.getElementById(ROOT_ID);
      if(!root){ root = document.createElement("div"); root.id = ROOT_ID; document.body.appendChild(root); }
      root.innerHTML = `<div class="drqa-box" style="max-width:680px;margin:0 auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif"><div id="drqa-messages" style="display:flex;flex-direction:column;gap:12px;padding:8px 0;"></div><div id="drqa-pills" style="display:flex;flex-wrap:wrap;gap:8px;padding:6px 0 12px;"></div><form id="drqa-form" style="display:flex;gap:8px;"><input id="drqa-input" type="text" placeholder="Ask about your shoulder…" autocomplete="off" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:8px;"><button type="submit" style="padding:10px 14px;border:0;border-radius:8px;background:#111827;color:#fff;cursor:pointer;">Ask</button></form><div style="margin-top:8px;font-size:12px;color:#6b7280;">Educational information only — not medical advice.</div></div>`;
      var msgs = root.querySelector("#drqa-messages");
      var pills = root.querySelector("#drqa-pills");
      var form = root.querySelector("#drqa-form");
      var input = root.querySelector("#drqa-input");
      function addMsg(text, who){
        var d = document.createElement("div");
        d.style.padding="10px 12px"; d.style.borderRadius="12px"; d.style.lineHeight="1.35";
        if(who==="user"){ d.style.background="#eef2ff"; d.style.alignSelf="flex-end"; }
        else { d.style.background="#f4f4f5"; }
        d.textContent = text; msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight;
      }
      function renderPills(arr){
        pills.innerHTML = "";
        (arr||[]).forEach(function(label){
          var b = document.createElement("button");
          b.type = "button"; b.textContent = label;
          b.style.border="1px solid #ddd"; b.style.borderRadius="999px"; b.style.padding="6px 10px";
          b.style.cursor="pointer"; b.style.fontSize="14px"; b.style.background="#fff";
          b.onclick = function(){ input.value = label; form.dispatchEvent(new Event("submit",{cancelable:true})); };
          pills.appendChild(b);
        });
      }
      async function ask(q){
        addMsg(q, "user"); input.value="";
        try{
          var res = await fetch(API + "/ask", {method: "POST",headers: {"Content-Type":"application/json"},body: JSON.stringify({ question: q, topic: TOPIC })});
          var data = await res.json();
          addMsg(data.answer, "bot"); if (data.practice_notes) addMsg(data.practice_notes, "bot");
          renderPills(data.suggestions || []);
        }catch(e){ addMsg("Sorry — something went wrong. Please try again.", "bot"); }
      }
      renderPills(["What is shoulder arthroscopy?","What are the risks?","What is recovery like?","When can I drive?"]);
      form.addEventListener("submit", function(ev){ ev.preventDefault(); var q = input.value.trim(); if(q) ask(q); });
    })();"""

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Patient Education Chat</title><style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#f9fafb;display:flex;justify-content:center;padding:40px;}</style></head><body><div id="drqa-root"></div><script src="/widget.js" defer></script></body></html>"""
