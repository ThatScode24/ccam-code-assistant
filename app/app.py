"""Gradio demo: free-text French act description -> top-5 CCAM codes.

Local:  .venv/bin/python app/app.py            (uses models/e5-base-ccam-v2 by default)
Spaces: set CCAM_MODEL to a Hugging Face model id, ship codes.jsonl next to this file.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # local repo layout
sys.path.insert(0, str(HERE))         # Spaces layout (ccam_ai/ copied next to app.py)
from ccam_ai.abbrev import expand  # noqa: E402
from ccam_ai.retrievers import BM25Retriever, DenseRetriever  # noqa: E402

MODEL = os.environ.get("CCAM_MODEL", str(HERE.parent / "models" / "e5-base-ccam-v2"))
CODES_PATH = next(p for p in (HERE / "codes.jsonl", HERE.parent / "data/processed/codes.jsonl")
                  if p.exists())

codes = [json.loads(line) for line in open(CODES_PATH, encoding="utf-8")]
by_code = {c["code"]: c for c in codes}
dense = DenseRetriever(codes, MODEL, "fine-tuned")
bm25 = BM25Retriever(codes)


def _table(hits):
    return [[c, by_code[c]["label"], by_code[c].get("chapter", ""), round(s, 3)] for c, s in hits]


def suggest(text: str):
    text = (text or "").strip()
    if not text:
        return [], []
    # abbreviations (PTH, JJ, FOGD...) are expanded for the model; BM25 stays a raw keyword baseline
    return _table(dense.search([expand(text)], k=5)[0]), _table(bm25.search([text], k=5)[0])


EXAMPLES = [
    "ablation d'un kyste sous-cutané du dos",
    "PTH G voie postérieure pour coxarthrose",
    "coelio pour appendicite aiguë",
    "pose de sonde JJ à droite sous AG",
    "exerese naevus joue 1cm suture simple",
    "échographie abdominale complète",
]
HEAD = ["Code", "Libellé CCAM", "Chapitre", "Score"]

with gr.Blocks(title="Assistant de codage CCAM") as demo:
    gr.Markdown(
        "# Assistant de codage CCAM\n"
        "Décrivez l'acte comme dans un compte rendu ; le modèle propose les 5 codes CCAM les "
        "plus probables. *Outil d'aide : le codage final reste sous la responsabilité du praticien.*"
    )
    inp = gr.Textbox(label="Description de l'acte", lines=2,
                     placeholder="ex. ablation d'un kyste sous-cutané du dos")
    btn = gr.Button("Proposer des codes", variant="primary")
    out = gr.Dataframe(headers=HEAD, label="Modèle fine-tuné (top 5)", wrap=True)
    with gr.Accordion("Comparaison : recherche par mots-clés (BM25)", open=False):
        out_bm25 = gr.Dataframe(headers=HEAD, wrap=True)
    gr.Examples(EXAMPLES, inp)
    btn.click(suggest, inp, [out, out_bm25])
    inp.submit(suggest, inp, [out, out_bm25])

if __name__ == "__main__":
    demo.launch()
