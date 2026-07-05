"""
main.py — Pessoa 4: API FastAPI que expõe a busca semântica + formatação por
LLM local (src/llm/answer_formatter.py) pro app Streamlit consumir.

IMPORTANTE — busca provisória: até as branches pessoa2-busca-leve e
pessoa3-busca-pesada serem mescladas aqui, não existe nenhum dos 4 métodos de
busca reais disponíveis nesta branch (eles vivem em src/search/, que ainda
está vazio por aqui). Pra não bloquear o desenvolvimento da API/LLM/app
esperando o merge, e pra não reinventar o trabalho de busca de ninguém,
usamos abaixo uma busca por palavra-chave BEM simples só como placeholder.
Assim que src/search/ (com BaseSemanticSearch) estiver disponível nesta
branch, troquem `_placeholder_search` por uma instância real (TfidfCosineSearch,
Word2VecAverageSearch ou HeavyRetriever) — a assinatura de retorno já segue o
mesmo formato (title, snippet, score) pra facilitar a troca.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm.answer_formatter import MovieHit, format_answer

BASE_DIR = Path(__file__).resolve().parents[2]
MOVIES_PARQUET = BASE_DIR / "data" / "processed" / "movies.parquet"

app = FastAPI(title="PAA Busca Semântica — API")

_MOVIES_DF: Optional[pd.DataFrame] = None


def _get_movies_df() -> pd.DataFrame:
    global _MOVIES_DF
    if _MOVIES_DF is None:
        if not MOVIES_PARQUET.exists():
            raise FileNotFoundError(
                f"'{MOVIES_PARQUET}' não encontrado. Rode "
                "src/preprocessing/preprocess.py primeiro (ver README)."
            )
        _MOVIES_DF = pd.read_parquet(MOVIES_PARQUET)
    return _MOVIES_DF


def _placeholder_search(query: str, top_k: int) -> list[MovieHit]:
    """Busca por contagem de palavras em comum — O(n) simples, só placeholder
    (ver aviso no topo do arquivo). NÃO é um método de busca semântica."""
    query_words = set(query.lower().split())

    df = _get_movies_df().copy()
    df["match_score"] = df["clean_summary"].apply(
        lambda text: len(query_words & set(text.split()))
    )
    top = df.sort_values("match_score", ascending=False).head(top_k)

    return [
        MovieHit(title=row.movie_name, snippet=row.clean_summary[:200], score=float(row.match_score))
        for row in top.itertuples()
    ]


class AskResponse(BaseModel):
    query: str
    answer: str
    movies: list[dict]


@app.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., min_length=1), top_k: int = 5):
    hits = _placeholder_search(q, top_k)
    answer = format_answer(q, hits)
    return {
        "query": q,
        "answer": answer,
        "movies": [{"title": h.title, "snippet": h.snippet, "score": h.score} for h in hits],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
