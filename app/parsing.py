import docx
from typing import List
import re

HEADING_RX = re.compile(r"^(chapter\s+\d+\.|faq|frequently asked questions|q:|question:)", re.I)

def _split_long(text: str, target: int = 400, overlap: int = 60) -> List[str]:
    # split by double-newlines first
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: List[str] = []
    for p in parts:
        if len(p) <= target:
            out.append(p)
        else:
            # sentence-level split; crude but effective
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sents:
                if not buf:
                    buf = s
                elif len(buf) + 1 + len(s) <= target:
                    buf = buf + " " + s
                else:
                    out.append(buf.strip())
                    # overlap tail to keep continuity
                    tail = buf[-overlap:] if len(buf) > overlap else buf
                    buf = (tail + " " + s).strip()
            if buf:
                out.append(buf.strip())
    return out

def read_docx_chunks(path: str) -> List[str]:
    doc = docx.Document(path)
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf: return
        text = " ".join(t.strip() for t in buf if t.strip())
        buf.clear()
        # split long text into smaller subchunks
        for sub in _split_long(text, target=420, overlap=60):
            if len(sub) >= 80:  # drop tiny fragments
                blocks.append(sub)

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        if HEADING_RX.search(t):
            flush()
            buf.append(t)
        else:
            buf.append(t)
    flush()
    return blocks
