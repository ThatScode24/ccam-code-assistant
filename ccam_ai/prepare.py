"""Build data/processed/codes.jsonl from the raw CCAM sources.

Sources (in data/raw/):
  * ameli CCAM export (*.xls / *.xlsx, any name containing "ccam")  -> full list, preferred
  * interhop-actes-ameli.csv (data.gouv.fr mirror, ~2 000 acts)      -> fallback
  * open-ccam-national.csv (ATIH)                                    -> national hospital volumes

Usage:  python -m ccam_ai.prepare
"""
from __future__ import annotations

import collections
import csv
import re

from .common import CODE_RE, CODES_FILE, RAW, write_jsonl

HIER_RE = re.compile(r"^\d{2}(\.\d{2}){0,5}$")


def _clean_label(s: str) -> str:
    s = " ".join(str(s).split())
    return s.lower() if s.isupper() else s


def load_interhop() -> list[dict]:
    path = RAW / "interhop-actes-ameli.csv"
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r["ccam"].strip()
            if not CODE_RE.match(code) or not r["label"].strip() or code in out:
                continue
            out[code] = {
                "code": code,
                "label": _clean_label(r["label"]),
                "chapter": r["chapterLabel"].strip(),
                "parent_id": r["parentCode"].strip(),
                "parent_label": r["parentLabel"].strip(),
            }
    return list(out.values())


def load_ameli_excel(path) -> list[dict]:
    """Generic parser for the ameli CCAM Excel export.

    Walks every row; rows whose first non-empty cell is a hierarchy number (01.02.03)
    update the current chapter/parent title, rows containing a 7-char CCAM code become
    acts whose label is the longest text cell on that row.
    """
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    out: dict[str, dict] = {}
    for df in sheets.values():
        titles: dict[int, str] = {}
        parent_id = ""
        for row in df.itertuples(index=False):
            cells = [" ".join(str(c).split()) for c in row if isinstance(c, str) and c.strip()]
            if not cells:
                continue
            first = cells[0]
            if HIER_RE.match(first) and len(cells) > 1:
                depth = first.count(".")
                titles = {d: t for d, t in titles.items() if d < depth}
                titles[depth] = _clean_label(cells[1])
                parent_id = first
                continue
            code = next((c for c in cells[:3] if CODE_RE.match(c)), None)
            if not code or code in out:
                continue
            texts = [c for c in cells if c != code and re.search(r"[A-Za-zÀ-ÿ]{3}", c)]
            if not texts:
                continue
            deepest = max(titles) if titles else None
            out[code] = {
                "code": code,
                "label": _clean_label(max(texts, key=len)),
                "chapter": titles.get(0, ""),
                "parent_id": parent_id,
                "parent_label": titles[deepest] if deepest not in (None, 0) else "",
            }
    return list(out.values())


def load_ccam_terminology(path) -> list[dict]:
    """CCAM terminology export (one row per concept: Code / Parent immédiat / Terme Principal /
    Synonyme...). Acts are the 7-char codes; their parent is a 'Categorie_xx.xx' concept."""
    import pandas as pd

    df = pd.read_excel(path, sheet_name=0, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    cat_rows = df[df["Code"].str.startswith("Categorie_")]
    cats = {code: re.sub(r"^[\d.]+\s+", "", _clean_label(term))  # strip "01.02.03 " prefix
            for code, term in zip(cat_rows["Code"], cat_rows["Terme Principal"])}
    out = {}
    for r in df[df["Code"].str.match(r"^[A-Z]{4}\d{3}$")].to_dict("records"):
        parent = r["Parent immédiat"].strip()
        chapter_id = parent[: len("Categorie_00")]
        out[r["Code"]] = {
            "code": r["Code"],
            "label": " ".join(r["Terme Principal"].split()),
            "short_label": " ".join(r["Synonyme"].split()),
            "chapter": cats.get(chapter_id, ""),
            "parent_id": parent,
            "parent_label": cats.get(parent, ""),
            "type": r.get("typeActe", "").replace("TypeActe_", ""),
        }
    return list(out.values())


def load_volumes(year: str = "2024") -> collections.Counter:
    vol: collections.Counter = collections.Counter()
    path = RAW / "open-ccam-national.csv"
    if not path.exists():
        return vol
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["annee"] == year:
                vol[r["acte"][:7]] += int(float(r["nb_actes"] or 0))
    return vol


def add_siblings(codes: list[dict], k: int = 8) -> None:
    """Siblings = acts sharing the same hierarchy node, else the same 4-letter prefix."""
    by_parent = collections.defaultdict(list)
    by_prefix = collections.defaultdict(list)
    for c in codes:
        if c["parent_id"]:
            by_parent[c["parent_id"]].append(c["code"])
        by_prefix[c["code"][:4]].append(c["code"])
    for c in codes:
        sib = [x for x in by_parent.get(c["parent_id"], []) if x != c["code"]]
        sib += [x for x in by_prefix[c["code"][:4]] if x != c["code"] and x not in sib]
        c["siblings"] = sib[:k]


def main() -> None:
    excel = sorted(p for p in RAW.glob("*.xls*") if "ccam" in p.name.lower())
    if excel:
        import pandas as pd

        header = pd.read_excel(excel[-1], sheet_name=0, nrows=0).columns.str.strip().tolist()
        loader = load_ccam_terminology if "Terme Principal" in header else load_ameli_excel
        codes = loader(excel[-1])
        source = excel[-1].name
    else:
        codes = load_interhop()
        source = "interhop-actes-ameli.csv (partial list, ~2 000 acts)"
    vol = load_volumes()
    for c in codes:
        c["volume_2024"] = vol.get(c["code"], 0)
    add_siblings(codes)
    codes.sort(key=lambda c: c["code"])
    write_jsonl(CODES_FILE, codes)
    tot = sum(vol.values()) or 1
    cov = sum(c["volume_2024"] for c in codes) / tot
    print(f"source: {source}")
    print(f"{len(codes)} acts -> {CODES_FILE}  (covers {cov:.1%} of 2024 hospital act volume)")


if __name__ == "__main__":
    main()
