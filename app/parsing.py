import docx
from typing import List
import re

HEADING_RX = re.compile(r"^(chapter\s+\d+\.|faq|frequently asked questions)", re.I)

def read_docx_chunks(path: str) -> List[str]:
    doc = docx.Document(path)
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            text = " ".join(t.strip() for t in buf if t.strip())
            if len(text) > 120:
                blocks.append(text)
            buf.clear()

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
