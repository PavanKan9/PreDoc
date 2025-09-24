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

SUGGESTION_SYSTEM = """You generate 3–5 SHORT, SAFE quick-reply chips for a patient-education chat.
Rules:
- ≤ 45 characters per chip.
- Educational only. No diagnosis or treatment.
- No drug names or dosages.
- Patient-friendly wording.
Return ONLY a JSON array of strings."""

def clean_chips(arr: List[str], k: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in arr:
        s = s.strip()
        if 0 < len(s) <= 45 and s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())
        if len(out) >= k:
            break
    return out

def gen_suggestions(question: str, answer: str, topic: str = "shoulder", k: int = 4) -> List[str]:
    try:
        prompt = f"Topic: {topic}\nUser question: {question}\nAnswer: {answer}\nReturn JSON array (3–5)."
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "system", "content": SUGGESTION_SYSTEM},{"role": "user", "content": prompt}],
        )
        arr = json.loads(r.choices[0].message.content)
        chips = clean_chips(arr, k)
        if chips: return chips
    except Exception:
        pass
    return DEFAULTS.get(topic, DEFAULTS["shoulder"])[:k]
