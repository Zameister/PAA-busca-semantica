"""
retriever_heavy.py — Método 3 (Pessoa 3): busca semântica com embeddings SBERT
(SentenceTransformer) + FAISS. Consome o índice gerado por indexer_heavy.py.

Complexidade
------------
Indexação: ver indexer_heavy.py.

Busca (retrieve), por query:
    1) Embedding da query: custo fixo do modelo SentenceTransformer,
       independente de n (mesmo custo por consulta seja o corpus com 1000 ou
       1 milhão de filmes).
    2) `self.index.search(q_emb, top_k)`: o índice carregado é o
       `IndexFlatIP` construído em indexer_heavy.py — ou seja, EXATO/
       exaustivo, não aproximado. A busca calcula o produto interno da query
       contra os n vetores do índice inteiros e depois seleciona os top_k
       maiores. Custo: O(n · dim) por consulta (dim = dimensão do embedding,
       384 pro modelo default `all-MiniLM-L6-v2`), mais um O(n log top_k)
       pequeno pra selecionar os melhores.

    => Isso é DIFERENTE do que os slides originais do grupo descreviam
       (HNSW — Hierarchical Navigable Small World). HNSW é uma estrutura de
       indexação APROXIMADA: organiza os vetores num grafo de navegação em
       camadas, permitindo busca em O(log n) esperado, ao custo de não
       garantir sempre os vizinhos mais próximos exatos (recall < 100%).
       O código atual usa `faiss.IndexFlatIP`, que é EXATO (sempre encontra
       os top-k reais, recall = 100% por construção) mas não reduz o "n" —
       a busca continua sendo O(n · dim), essencialmente o mesmo formato de
       custo do Método 1 (TF-IDF), só que com embeddings densos (SBERT) no
       lugar de esparsos (TF-IDF).
       Trade-off: exato-e-mais-lento (Flat, o que está implementado) vs
       aproximado-e-mais-rápido (HNSW, o que os slides descreviam). Pra virar
       a versão HNSW de fato, bastaria trocar `faiss.IndexFlatIP(dim)` por
       `faiss.IndexHNSWFlat(dim, M)` em indexer_heavy.py — a interface de
       `retrieve()` não mudaria.
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise ImportError("Install sentence-transformers to use retriever: pip install sentence-transformers")

try:
    import faiss
except Exception:
    raise ImportError("Install faiss-cpu or faiss-gpu to use FAISS: pip install faiss-cpu")


def _to_jsonable(value):
    """metadata.parquet guarda colunas cruas de movies.parquet (tokens como
    ndarray, ids como np.int64, etc.) que o FastAPI/pydantic não sabe serializar
    direto -- converte pros tipos nativos do Python antes de devolver na API."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class HeavyRetriever:
    def __init__(self, index_dir: str = "artifacts/heavy_index", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", use_gpu: bool = False):
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")

        self.model = SentenceTransformer(model_name, device="cuda" if use_gpu else "cpu")

        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "metadata.parquet"
        self.index_path = self.index_dir / "index.faiss"

        self.metadata = pd.read_parquet(self.meta_path)

        self.index = faiss.read_index(str(self.index_path))

    def _embed_query(self, text: str):
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def retrieve(self, query: str, top_k: int = 10):
        q_emb = self._embed_query(query).astype('float32')
        q_emb = np.expand_dims(q_emb, axis=0)
        D, I = self.index.search(q_emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()

        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            row = {k: _to_jsonable(v) for k, v in self.metadata.iloc[idx].to_dict().items()}
            row.update({"score": float(score), "_idx": int(idx)})
            results.append(row)

        return results


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="artifacts/heavy_index")
    parser.add_argument("--q", required=True)
    args = parser.parse_args()

    r = HeavyRetriever(args.index_dir)
    pprint(r.retrieve(args.q, top_k=10))
