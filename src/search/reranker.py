"""
reranker.py — etapa de reranking (Pessoa 3) com Cross-Encoder, aplicada sobre
os candidatos já filtrados pelo retriever_heavy.py (Método 3).

Complexidade
------------
Custo por par (query, candidato): diferente do bi-encoder usado no retriever
(que embeda query e documentos SEPARADAMENTE, permitindo pré-computar os
embeddings dos documentos uma única vez na indexação), o Cross-Encoder recebe
query e documento CONCATENADOS numa única entrada e roda uma passada completa
do transformer sobre esse par. Não há nada pra pré-computar: o custo do
encoder é pago de novo a cada par, sempre — é por isso que o Cross-Encoder é
mais preciso (a atenção do transformer olha os dois textos juntos) mas muito
mais caro por comparação do que o bi-encoder.

Custo total do rerank: O(K) passadas pelo modelo, onde K = `len(candidates)`
— não é o corpus inteiro (n). Em heavy_search_api.py, quem chama esta função
passa `candidates = retriever.retrieve(q, top_k=top_k*3)`, ou seja, K = 3×
top_k pedido pelo usuário. O reranking sempre opera sobre um número pequeno e
fixo de candidatos já pré-filtrados pela busca dos Métodos 3 (nunca sobre n) —
é esse "funil" (retrieve barato sobre todo o corpus, rerank caro só nos
poucos finalistas) que torna o Cross-Encoder viável apesar de custar muito
mais por item do que o bi-encoder.

O modelo é carregado uma única vez por `model_name` (`_get_cross_encoder`
abaixo) e reaproveitado entre chamadas — o custo de carregamento só é pago
na primeira vez; nas chamadas seguintes o custo é só o O(K) de inferência
descrito acima.
"""

from typing import List, Dict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

_cross_encoders: dict = {}  # model_name -> instância já carregada


def _get_cross_encoder(model_name: str):
    if model_name not in _cross_encoders:
        _cross_encoders[model_name] = CrossEncoder(model_name)
    return _cross_encoders[model_name]


def rerank_with_cross_encoder(query: str, candidates: List[Dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
    """Reordena `candidates` (lista de dicts contendo `clean_summary`) usando um CrossEncoder.

    Se o CrossEncoder não estiver disponível, retorna os candidatos sem alteração.
    """
    if CrossEncoder is None:
        return candidates

    texts = [c.get("clean_summary", "") for c in candidates]
    pairs = [[query, t] for t in texts]
    model = _get_cross_encoder(model_name)
    scores = model.predict(pairs, batch_size=batch_size)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return ranked
