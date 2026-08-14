"""
retriever.py — Hybrid retrieval (vector + BM25) and contextual compression.
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import RETRIEVAL_K, FINAL_K, SIM_THRESHOLD
from src.pipeline.embeddings import FastEmbeddings


class BM25Retriever:
    """Keyword-based retriever built from the pre-saved chunks.json."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([d.page_content.lower().split() for d in docs])

    def invoke(self, query: str, k: int = RETRIEVAL_K) -> list[Document]:
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[-k:][::-1]
        return [self.docs[i] for i in top_idx if scores[i] > 0]

    @classmethod
    def from_json(cls, path: Path) -> "BM25Retriever | None":
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=c["content"], metadata=c["metadata"])
                for c in data]
        return cls(docs)


def compress(query: str,
             docs: list[Document],
             embeddings: FastEmbeddings) -> list[Document]:
    """
    Contextual compression: score each doc by cosine similarity to the query,
    keep only those above SIM_THRESHOLD, return top FINAL_K.
    """
    if not docs:
        return docs
    q  = np.array(embeddings.embed_query(query))
    d  = np.array(embeddings.embed_documents([x.page_content for x in docs]))
    qn = q / (np.linalg.norm(q) + 1e-9)
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    sims   = dn @ qn
    ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)
    return [doc for doc, s in ranked if s >= SIM_THRESHOLD][:FINAL_K]
