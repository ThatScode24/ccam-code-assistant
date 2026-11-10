"""Retrievers over the CCAM catalogue: BM25 keyword baseline and dense bi-encoder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import code_document, tokenize_fr


class BM25Retriever:
    name = "BM25 (keyword search)"

    def __init__(self, codes: list[dict]):
        from rank_bm25 import BM25Okapi

        self.codes = [c["code"] for c in codes]
        self.bm25 = BM25Okapi([tokenize_fr(code_document(c)) for c in codes])

    def search(self, queries: list[str], k: int = 10) -> list[list[tuple[str, float]]]:
        out = []
        for q in queries:
            s = self.bm25.get_scores(tokenize_fr(q))
            idx = np.argsort(-s)[:k]
            out.append([(self.codes[i], float(s[i])) for i in idx])
        return out


def prefixes_for(model_name_or_path: str) -> tuple[str, str]:
    cfg = Path(model_name_or_path) / "ccam_prefix.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return d["query"], d["passage"]
    if "e5" in str(model_name_or_path).lower():
        return "query: ", "passage: "
    return "", ""


class DenseRetriever:
    def __init__(self, codes: list[dict], model: str, name: str | None = None, batch_size: int = 128):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = name or model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        self.model.max_seq_length = 128
        self.qp, self.pp = prefixes_for(model)
        self.batch_size = batch_size
        self.codes = [c["code"] for c in codes]
        self.emb = self.model.encode(
            [self.pp + code_document(c) for c in codes], batch_size=batch_size,
            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False,
        )

    def search(self, queries: list[str], k: int = 10) -> list[list[tuple[str, float]]]:
        q = self.model.encode([self.qp + x for x in queries], batch_size=self.batch_size,
                              normalize_embeddings=True, convert_to_tensor=True,
                              show_progress_bar=False)
        scores = q @ self.emb.T
        top = scores.topk(k, dim=1)
        vals, idx = top.values.cpu().numpy(), top.indices.cpu().numpy()
        return [[(self.codes[j], float(v)) for j, v in zip(row_i, row_v)]
                for row_i, row_v in zip(idx, vals)]
