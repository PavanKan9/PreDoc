from typing import List
import json
from openai import OpenAI

client = OpenAI()

DEFAULTS = {
    "shoulder": [
        "What is shoulder arthroscopy?",
        "What are the risks?",
        "What is recovery like?",
        "When can I drive?",
    ],
}

SUGGESTION_SYSTEM = """You generate 3–5 SHORT, SAFE follow-up QUESTIONS for a patient-education chat.

Rules:
- Each chip must be phrased as a QUESTION (e.g., "What are the risks of this surgery?").
- ≤ 45 characters. Patient-friendly, sentence case.
- Base chips on the user’s last question AND the assistant’s last answer.
- The goal is to let the patient dive deeper into related subtopics.
- Avoid duplication with previous chips (use 'avoid' list).
- Do NOT diagnose or prescribe treatment. No drug names/dosages.
- If red flags (fever + swelling, deformity, numbness/weakness) → include urgent chip: 
  "Should I seek urgent care?".
- Output ONLY a JSON array of question strings.
"""

def clean_chips(arr: List[str], k: int, avoid: List[str]) -> List[str]:
    seen = {s.lower().strip() for s in (avoid or [])}
    out: List[str] = []
    for s in arr or []:
        s = s.strip()
        if 0 < len(s) <= 45 and s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())
        if len(out) >= k:
            break
    return out

def gen_suggestions(question: str, answer: str, topic: str = "shoulder", k: int = 4, avoid: List[str] = []) -> List[str]:
    # 1) Try model
    try:
        prompt = json.dumps({
            "topic": topic,
            "question": question,
            "answer": answer,
            "avoid": avoid,
            "k": k
        })
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": SUGGESTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        arr = json.loads(r.choices[0].message.content)
        chips = clean_chips(arr, k, avoid)
        if chips:
            return chips
    except Exception:
        pass

    # 2) Fallback defaults
    return clean_chips(DEFAULTS.get(topic, DEFAULTS["shoulder"]), k, avoid)
