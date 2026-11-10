"""Spot check on short, doctor-style queries written by hand (by the project author, NOT real
clinical data and NOT a benchmark: the abbreviation list was written with these in mind).
It is a sanity check that complements the synthetic test sets; the real measure is a gold set
of real descriptions (data/eval/gold.csv).

Usage:  python -m ccam_ai.spotcheck [model_dir ...]
"""
from __future__ import annotations

import sys

from .abbrev import Expanded
from .common import MODELS, load_codes
from .retrievers import BM25Retriever, DenseRetriever

# query -> acceptable codes
QUERIES: list[tuple[str, set[str]]] = [
    ("PTH G voie postérieure pour coxarthrose", {"NEKA020"}),
    ("PTG D sur gonarthrose", {"NFKA007", "NFKA008"}),
    ("pose de sonde JJ à droite sous AG", {"JCLE002"}),
    ("exerese naevus joue 1cm suture simple", {"QZFA036"}),
    ("ablation d'un kyste sous-cutané du dos", {"QZFA002"}),
    ("kyste sébacé du dos 2 cm", {"QZFA002"}),
    ("exérèse lipome épaule 5 cm", {"QZFA007"}),
    ("chole coelio", {"HMFC004"}),
    ("coelio pour appendicite aiguë", {"HHFA016"}),
    ("écho abdo", {"ZCQM008", "ZCQM001", "ZCQM005"}),
    ("FOGD", {"HEQE002"}),
    ("RTUP", {"JGFE023", "JGFE365", "JGFA015"}),
    ("TAVI par voie fémorale", {"DBLF001"}),
    ("cataracte OD phaco + implant", {"BFGA004"}),
    ("IVG par aspiration", {"JNJD002"}),
    ("pose DIU", {"JKLD001"}),
    ("coloscopie totale", {"HHQE002", "HHQE005"}),
    ("ETT", {"DZQM006"}),
    ("IRM cérébrale sans injection", {"ACQN001"}),
    ("scanner abdomino-pelvien injecté", {"ZCQH001"}),
    ("hernie inguinale G sous coelio avec plaque", {"LMMC002"}),
    ("canal carpien", {"AHPA009", "AHPC001"}),
    ("ménisectomie interne arthroscopique genou", {"NFFC004"}),
    ("ponction lombaire", {"AFHB002"}),
    ("PL", {"AFHB002"}),
    ("ECG", {"DEQP003"}),
    ("Holter ECG 24h", {"DEQP005"}),
    ("pose de port-à-cath", {"EBLA003"}),
    ("fibro bronchique avec LBA", {"GEQE004"}),
    ("BOM", {"FDHB002"}),
]


def rank_of(hits, gold) -> int | None:
    for i, (c, _) in enumerate(hits, 1):
        if c in gold:
            return i
    return None


def main() -> None:
    codes = load_codes()
    models = sys.argv[1:] or sorted(str(p) for p in MODELS.glob("*") if (p / "ccam_prefix.json").exists())
    bm25 = BM25Retriever(codes)
    retrievers = [bm25, Expanded(bm25)]
    for m in models:
        d = DenseRetriever(codes, m, m.rstrip("/").split("/")[-1])
        retrievers += [d, Expanded(d)]
    qs = [q for q, _ in QUERIES]
    table = {r.name: [rank_of(h, g) for h, (_, g) in zip(r.search(qs, k=10), QUERIES)]
             for r in retrievers}
    short = {n: n.replace("BM25 (keyword search)", "bm25").replace(" + abbrev", "+ab")
             for n in table}
    print(f"{'query':45s} " + " ".join(f"{short[n][:14]:>14s}" for n in table))
    for i, q in enumerate(qs):
        print(f"{q[:45]:45s} " + " ".join(f"{(table[n][i] or '-'):>14}" for n in table))
    print(f"{'top-5 hits / ' + str(len(qs)):45s} "
          + " ".join(f"{sum(1 for x in table[n] if x and x <= 5):>14}" for n in table))


if __name__ == "__main__":
    main()
