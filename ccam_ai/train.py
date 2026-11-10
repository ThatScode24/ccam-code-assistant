"""Fine-tune a sentence-embedding model to map doctor descriptions -> CCAM label text.

Contrastive training (MultipleNegativesRankingLoss): every other code in the batch is a
negative, plus one *hard* negative per pair taken from the target's sibling acts (the codes
that are easiest to confuse with it).

Usage:
  python -m ccam_ai.train --base intfloat/multilingual-e5-base --out models/e5-base-ccam
  python -m ccam_ai.train --base dangvantuan/sentence-camembert-base --out models/camembert-ccam
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import PROC, code_document, load_codes, read_jsonl
from .retrievers import prefixes_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--no-hard-negatives", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    ap.add_argument("--train-file", default=str(PROC / "train.jsonl"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    rng = random.Random(0)
    codes = {c["code"]: c for c in load_codes()}
    qp, pp = prefixes_for(args.base)
    rows = {"anchor": [], "positive": []}
    if not args.no_hard_negatives:
        rows["negative"] = []
    for r in read_jsonl(Path(args.train_file)):
        c = codes[r["code"]]
        rows["anchor"].append(qp + r["query"])
        rows["positive"].append(pp + code_document(c))
        if "negative" in rows:
            sibs = [s for s in c.get("siblings", []) if s in codes]
            neg = codes[rng.choice(sibs)] if sibs else codes[rng.choice(list(codes))]
            rows["negative"].append(pp + code_document(neg))
    train = Dataset.from_dict(rows).shuffle(seed=0)
    print(f"{len(train)} training pairs from {len({*rows['positive']})} codes")

    model = SentenceTransformer(args.base)
    model.max_seq_length = 128
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=0.1,  # float = ratio in transformers v5
        bf16=torch.cuda.is_available(),
        max_steps=args.max_steps,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # same code twice in a batch = false negative
        save_strategy="no",
        logging_steps=50,
        logging_first_step=True,
        disable_tqdm=True,  # plain loss lines in nohup logs instead of \r progress bars
        report_to="none",
        seed=0,
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=train, loss=loss).train()
    model.save(args.out)
    (Path(args.out) / "ccam_prefix.json").write_text(json.dumps({"query": qp, "passage": pp}))
    print("saved", args.out)


if __name__ == "__main__":
    main()
