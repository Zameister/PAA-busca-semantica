"""
indexer_heavy.py — Método 3 (Pessoa 3): embeddings densos via SentenceTransformer
(SBERT) + índice FAISS. Constrói os embeddings de todo o corpus e monta o
índice usado depois por retriever_heavy.py.

Complexidade
------------
Indexação (build_index):
    1) Encoding das sinopses pelo SentenceTransformer: uma passada do modelo
       por batch, n documentos no total. Cada passada tem custo fixo do
       transformer (depende do tamanho do texto e do modelo, não do tamanho
       do corpus) — é o termo mais caro da indexação, bem mais pesado que o
       TF-IDF (Método 1) ou o treino do Word2Vec (Método 2), mas paralelizável
       em batch (e em GPU, se `use_gpu=True`).
    2) `faiss.IndexFlatIP(dim)` + `index.add(embeddings)`: "Flat" significa
       que o índice NÃO tem nenhuma estrutura de indexação de fato — é só um
       array denso com os n vetores (dimensão `dim`) guardados lado a lado.
       `add()` é O(n·dim) (cópia dos vetores), sem clustering, sem grafo, sem
       quantização.

Busca (ver retriever_heavy.py): como o índice é Flat, cada consulta compara
contra os n vetores inteiros — busca EXAUSTIVA, não aproximada. Ver a nota
detalhada em retriever_heavy.py sobre a diferença em relação ao HNSW citado
nos slides originais do grupo.
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise ImportError("Install sentence-transformers to run the heavy indexer: pip install sentence-transformers")

try:
    import faiss
except Exception as e:  # pragma: no cover
    raise ImportError("Install faiss-cpu or faiss-gpu to use FAISS: pip install faiss-cpu")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_index(
    movies_parquet: str,
    out_dir: str = "artifacts/heavy_index",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    use_gpu: bool = False,
):
    """Constrói embeddings e índice FAISS a partir de `movies_parquet`.

    Salva em `out_dir` os artefatos:
      - `embeddings.npy` (float32)
      - `metadata.parquet` (pandas)
      - `index.faiss` (FAISS index)
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    df = pd.read_parquet(movies_parquet)
    # Campos esperados: wiki_movie_id, movie_name, clean_summary
    text_col = "clean_summary"
    if text_col not in df.columns:
        raise ValueError(f"Parquet must contain column '{text_col}'")

    model = SentenceTransformer(model_name, device="cuda" if use_gpu else "cpu")

    n = len(df)
    dim = model.get_sentence_embedding_dimension()

    embeddings = np.zeros((n, dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_texts = df[text_col].iloc[start:end].astype(str).tolist()
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings[start:end] = emb

    # Normalizar para inner-product (cosine)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Salvar embeddings e metadata
    np.save(out_path / "embeddings.npy", embeddings)
    df.reset_index(drop=True).to_parquet(out_path / "metadata.parquet", index=False)

    # Construir índice FAISS (IndexFlatIP para similaridade por produto interno)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_path / "index.faiss"))

    return {
        "out_dir": str(out_path),
        "n_items": n,
        "dim": dim,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to movies.parquet")
    parser.add_argument("--out", default="artifacts/heavy_index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    print("Building index... this may take a while")
    info = build_index(args.input, args.out, args.model, args.batch_size, args.use_gpu)
    print("Done:", info)
