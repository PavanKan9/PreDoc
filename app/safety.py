import re
from typing import List, Dict, Any

ER_PATTERNS = [
    r"\b(open|compound)\s+fracture\b",
    r"\bloss of pulse\b|\b(cold|blue)\s+(hand|foot)\b",
    r"\buncontrolled bleeding\b",
    r"\b(severe|sudden)\s+chest pain\b|\bshortness of breath\b",
]
UC_PATTERNS = [
    r"\bfever\b.*\b(joint|shoulder|arm)\b",
    r"\binability to (bear weight|move)\b|\bcannot (move|lift)\b",
    r"\b(new|worsening)\s+numbness\b|\bweakness\b|\btingling\b",
    r"\b(red|hot)\s+swollen\s+joint\b|\bsuspected infection\b",
    r"\bdeform(ity|ed)\b|\bpop\s+heard\b",
    r"\bsevere night pain\b|\bpain at rest\b(?!.*years)",
]

def triage_flags(text: str) -> Dict[str, Any]:
    t = text.lower()
    reasons: List[str] = []
    if any(re.search(p, t) for p in ER_PATTERNS):
        reasons = [p for p in ER_PATTERNS if re.search(p, t)]
        return {"triage": "er", "reasons": reasons}
    if any(re.search(p, t) for p in UC_PATTERNS):
        reasons = [p for p in UC_PATTERNS if re.search(p, t)]
        return {"triage": "urgent_care", "reasons": reasons}
    return {"triage": "ok", "reasons": []}
