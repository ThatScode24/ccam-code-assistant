"""Split generated queries into train / val / test sets.

* test_unseen: ALL queries of ~10% of codes that are never seen in training.
  This is the honest number: it measures generalisation to codes the model never saw.
* test_seen / val: a few held-out queries per training code. Optimistic, because they come
  from the same LLM call as the training queries for that code.

The retrieval index always contains every code, held-out ones included.

Usage:  python -m ccam_ai.split
"""
from __future__ import annotations

import collections
import random

from .common import JARGON_FILE, PROC, QUERIES_FILE, load_codes, read_jsonl, write_jsonl

SEED = 13
HELDOUT_FRAC = 0.10
N_TEST, N_VAL = 2, 1


def main() -> None:
    rows = read_jsonl(QUERIES_FILE)
    by_code = collections.defaultdict(list)
    for r in rows:
        by_code[r["code"]].append(r)
    rng = random.Random(SEED)
    codes = sorted(by_code)
    heldout = set(rng.sample(codes, max(1, int(len(codes) * HELDOUT_FRAC))))

    split = {"train": [], "val": [], "test_seen": [], "test_unseen": []}
    for code in codes:
        qs = by_code[code][:]
        rng.shuffle(qs)
        if code in heldout:
            split["test_unseen"] += qs
            continue
        split["test_seen"] += qs[:N_TEST]
        split["val"] += qs[N_TEST : N_TEST + N_VAL]
        split["train"] += qs[N_TEST + N_VAL :]
    # official abbreviated labels ("exérèse T. intraparench cervelet craniot.") are realistic
    # coder shorthand: extra training queries, for training codes only
    for c in load_codes():
        if c["code"] in by_code and c["code"] not in heldout and c.get("short_label"):
            split["train"].append({"code": c["code"], "query": c["short_label"]})
    # 2nd-pass jargon queries (acronyms, named conditions). Added after the rng calls above so
    # the original splits are unchanged and v1/v2 models are compared on identical test sets.
    # Held-out codes' jargon queries become an extra, harder unseen-code test set.
    if JARGON_FILE.exists():
        split["test_unseen_jargon"] = []
        for r in read_jsonl(JARGON_FILE):
            if r["code"] in heldout:
                split["test_unseen_jargon"].append(r)
            elif r["code"] in by_code:
                split["train"].append(r)
    for name, part in split.items():
        write_jsonl(PROC / f"{name}.jsonl", part)
        print(f"{name:12s} {len(part):6d} queries  {len({r['code'] for r in part}):5d} codes")


if __name__ == "__main__":
    main()
