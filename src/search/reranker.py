from typing import List, Dict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


def rerank_with_cross_encoder(query: str, candidates: List[Dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
    """Reordena `candidates` (lista de dicts contendo `clean_summary`) usando um CrossEncoder.

    Se o CrossEncoder não estiver disponível, retorna os candidatos sem alteração.
    """
    if CrossEncoder is None:
        # cross-encoder não instalado; fallback
        return candidates

    texts = [c.get("clean_summary", "") for c in candidates]
    pairs = [[query, t] for t in texts]
    model = CrossEncoder(model_name)
    scores = model.predict(pairs, batch_size=batch_size)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return ranked
