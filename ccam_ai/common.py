"""Shared paths, I/O helpers and text normalisation."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
EVAL = ROOT / "data" / "eval"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

CODES_FILE = PROC / "codes.jsonl"          # one CCAM act per line
QUERIES_FILE = PROC / "queries.jsonl"      # LLM-generated doctor-style descriptions
JARGON_FILE = PROC / "queries_jargon.jsonl"  # 2nd pass: acronyms + named conditions
SPLITS_FILE = PROC / "splits.json"         # held-out codes + query split
GOLD_FILE = EVAL / "gold.csv"              # optional hand-labelled real descriptions

CODE_RE = re.compile(r"^[A-Z]{4}\d{3}$")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_codes() -> list[dict]:
    return read_jsonl(CODES_FILE)


def code_document(c: dict) -> str:
    """Text that represents a CCAM code on the index side of retrieval."""
    doc = c["label"]
    if c.get("parent_label") and c["parent_label"].lower() not in doc.lower():
        doc += f" ({c['parent_label']})"
    return doc


# --- normalisation for keyword search -------------------------------------------------

_STOP = set(
    """a au aux avec ce ces d dans de des du en et l la le les leur lui ma mais me
    meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta
    te tes toi ton tu un une vos votre vous y sans sous entre chez lors plus ou sous apres
    avant via""".split()
)


def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def tokenize_fr(s: str) -> list[str]:
    """Lowercase, accent-fold, drop stopwords, crude plural stemming."""
    toks = re.findall(r"[a-z0-9]+", strip_accents(s.lower()))
    out = []
    for t in toks:
        if t in _STOP or len(t) == 1:
            continue
        if len(t) > 4 and t[-1] in "sx":
            t = t[:-1]
        out.append(t)
    return out
