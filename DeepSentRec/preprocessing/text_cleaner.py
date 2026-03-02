"""
Lightweight text cleaning aligned with your preprocessing description:
- lowercase
- remove special chars (conservative)
- expand common contractions (minimal)
- strip whitespace

Tokenization is handled by transformer tokenizer.
"""

from __future__ import annotations
import re

_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    for k, v in _CONTRACTIONS.items():
        s = s.replace(k, v)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
