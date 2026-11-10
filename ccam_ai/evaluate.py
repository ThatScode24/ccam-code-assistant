"""Compare retrievers: top-k accuracy and MRR on each test set.

Usage:
  python -m ccam_ai.evaluate                         # BM25 + zero-shot + every model in models/
  python -m ccam_ai.evaluate --models models/e5-base-ccam
Writes results/metrics.md and results/metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import json

from .common import GOLD_FILE, MODELS, PROC, RESULTS, load_codes, read_jsonl
from .abbrev import Expanded
from .retrievers import BM25Retriever, DenseRetriever

KS = (1, 3, 5, 10)
ZERO_SHOT = ["intfloat/multilingual-e5-base", "dangvantuan/sentence-camembert-base"]


def load_sets() -> dict[str, list[dict]]:
    sets = {}
    for name in ("test_seen", "test_unseen", "test_unseen_jargon"):
        p = PROC / f"{name}.jsonl"
        if p.exists():
            sets[name] = read_jsonl(p)
    if GOLD_FILE.exists():
        with open(GOLD_FILE, encoding="utf-8") as f:
            sets["gold"] = [{"query": r["query"], "code": r["code"].strip().upper()}
                            for r in csv.DictReader(f) if r.get("query") and r.get("code")]
    return {k: v for k, v in sets.items() if v}


def score(retriever, rows: list[dict]) -> dict[str, float]:
    hits = retriever.search([r["query"] for r in rows], k=max(KS))
    m = {f"acc@{k}": 0.0 for k in KS}
    m["mrr@10"] = 0.0
    for r, h in zip(rows, hits):
        ranked = [c for c, _ in h]
        if r["code"] in ranked:
            rank = ranked.index(r["code"]) + 1
            m["mrr@10"] += 1 / rank
            for k in KS:
                m[f"acc@{k}"] += rank <= k
    return {k: v / len(rows) for k, v in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="fine-tuned model dirs (default: every dir in models/)")
    ap.add_argument("--no-zero-shot", action="store_true")
    args = ap.parse_args()

    codes = load_codes()
    sets = load_sets()
    known = {c["code"] for c in codes}
    if "gold" in sets:
        missing = sum(r["code"] not in known for r in sets["gold"])
        print(f"gold set: {len(sets['gold'])} rows, {missing} with a code absent from the index")

    finetuned = args.models if args.models is not None else sorted(
        str(p) for p in MODELS.glob("*") if (p / "ccam_prefix.json").exists())
    candidates = [("bm25", None)]
    if not args.no_zero_shot:
        candidates += [(m, f"{m} (zero-shot)") for m in ZERO_SHOT]
    candidates += [(m, f"{m.rstrip('/').split('/')[-1]} (fine-tuned)") for m in finetuned]

    results = {}
    for model, name in candidates:
        base = BM25Retriever(codes) if model == "bm25" else DenseRetriever(codes, model, name)
        # zero-shot models: raw queries only; BM25 + fine-tuned: also with abbreviation expansion
        variants = [base] if "zero-shot" in base.name else [base, Expanded(base)]
        for r in variants:
            results[r.name] = {s: score(r, rows) for s, rows in sets.items()}
            print(r.name, json.dumps({s: round(v["acc@5"], 3) for s, v in results[r.name].items()}))
        del base, variants

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    lines = [f"Index: {len(codes)} CCAM codes.\n"]
    for s, rows in sets.items():
        lines += [f"\n### {s} ({len(rows)} queries, {len({r['code'] for r in rows})} codes)\n",
                  "| method | " + " | ".join(f"acc@{k}" for k in KS) + " | MRR@10 |",
                  "|---" * (len(KS) + 2) + "|"]
        for name, res in results.items():
            v = res[s]
            lines.append(f"| {name} | " + " | ".join(f"{v[f'acc@{k}']:.1%}" for k in KS)
                         + f" | {v['mrr@10']:.3f} |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
