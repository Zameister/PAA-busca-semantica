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
