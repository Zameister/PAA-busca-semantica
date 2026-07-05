"""Script utilitário para pré-computar embeddings e índice FAISS.

Uso:
    python scripts/precompute_embeddings.py --input data/processed/movies.parquet --out artifacts/heavy_index
"""
import sys
from pathlib import Path
import argparse

# raiz do repo (pai de scripts/) precisa estar no sys.path pro import "src.search"
# funcionar quando este arquivo é rodado direto (python scripts/precompute_embeddings.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search.indexer_heavy import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to movies.parquet")
    parser.add_argument("--out", default="artifacts/heavy_index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    info = build_index(args.input, args.out, args.model, args.batch_size, args.use_gpu)
    print("Index built:", info)


if __name__ == "__main__":
    main()
