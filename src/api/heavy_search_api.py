from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

from src.search.retriever_heavy import HeavyRetriever
from src.search.reranker import rerank_with_cross_encoder

app = FastAPI(title="Heavy Search API")

_RETRIEVER: Optional[HeavyRetriever] = None


class SearchResponse(BaseModel):
    results: List[dict]


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HeavyRetriever()
    return _RETRIEVER


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), top_k: int = 10):
    r = get_retriever()
    candidates = r.retrieve(q, top_k=top_k * 3)
    ranked = rerank_with_cross_encoder(q, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return {"results": ranked[:top_k]}
