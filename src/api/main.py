"""
main.py — Pessoa 4: API FastAPI que expõe a busca semântica + formatação por
LLM local (src/llm/answer_formatter.py) pro app Streamlit consumir.

Busca usada: Método 3 (SBERT + FAISS, retriever_heavy.py) com reranking por
Cross-Encoder (reranker.py), ambos da Pessoa 3 — é o método com melhor
trade-off qualidade/velocidade do grupo (ver EXPLICACAO_PARTES.md). Precisa
do índice pré-computado em `artifacts/heavy_index/` — gere com:
    python scripts/precompute_embeddings.py --input data/processed/movies.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.answer_formatter import MovieHit, format_answer
from src.search.retriever_heavy import HeavyRetriever
from src.search.reranker import rerank_with_cross_encoder

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "artifacts" / "heavy_index"

app = FastAPI(title="PAA Busca Semântica — API")

_RETRIEVER: Optional[HeavyRetriever] = None


def _get_retriever() -> HeavyRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"'{INDEX_DIR}' não encontrado. Gere o índice primeiro com "
                "scripts/precompute_embeddings.py (ver README)."
            )
        _RETRIEVER = HeavyRetriever(index_dir=str(INDEX_DIR))
    return _RETRIEVER


def _real_search(query: str, top_k: int) -> list[MovieHit]:
    """Busca real: retrieve (SBERT+FAISS) sobre 3x top_k candidatos, depois
    rerank (Cross-Encoder) só nesses poucos candidatos — ver a complexidade
    documentada em retriever_heavy.py e reranker.py."""
    retriever = _get_retriever()
    candidates = retriever.retrieve(query, top_k=top_k * 3)
    ranked = rerank_with_cross_encoder(query, candidates)

    return [
        MovieHit(title=c["movie_name"], snippet=c["clean_summary"][:200], score=c.get("rerank_score", c["score"]))
        for c in ranked[:top_k]
    ]


class AskResponse(BaseModel):
    query: str
    answer: str
    movies: list[dict]


@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=1), top_k: int = 5):
    hits = _real_search(q, top_k)
    answer = format_answer(q, hits)
    return {
        "query": q,
        "answer": answer,
        "movies": [{"title": h.title, "snippet": h.snippet, "score": h.score} for h in hits],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
