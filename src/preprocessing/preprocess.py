"""
Pre-processamento do CMU Movie Summary Corpus.

Baixa o dataset e junta os resumos de enredo com os metadados dos filmes.
"""

import ast
import tarfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ARCHIVE_PATH = RAW_DIR / "MovieSummaries.tar.gz"
EXTRACT_DIR = RAW_DIR / "MovieSummaries"

METADATA_COLUMNS = [
    "wiki_movie_id",
    "freebase_movie_id",
    "movie_name",
    "release_date",
    "box_office_revenue",
    "runtime",
    "languages",
    "countries",
    "genres",
]


def download_dataset():
    if ARCHIVE_PATH.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"baixando dataset de {DATASET_URL} ...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()
    with open(ARCHIVE_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def extract_dataset():
    if EXTRACT_DIR.exists():
        return
    print("extraindo arquivos...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DIR, filter="data")


def load_plot_summaries():
    path = EXTRACT_DIR / "plot_summaries.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["wiki_movie_id", "summary"])
    return df


def load_metadata():
    path = EXTRACT_DIR / "movie.metadata.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=METADATA_COLUMNS)
    return df


def extract_names(freebase_dict_str):
    """Campos de gênero/idioma/país vêm como string de dict {freebase_id: nome}."""
    try:
        d = ast.literal_eval(freebase_dict_str)
        return list(d.values())
    except (ValueError, SyntaxError):
        return []


def preprocess():
    download_dataset()
    extract_dataset()

    summaries = load_plot_summaries()
    metadata = load_metadata()

    df = summaries.merge(metadata, on="wiki_movie_id", how="inner")

    df["genres"] = df["genres"].apply(extract_names)
    df["languages"] = df["languages"].apply(extract_names)
    df["countries"] = df["countries"].apply(extract_names)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "movies.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{len(df)} filmes processados -> {out_path}")
    return df


if __name__ == "__main__":
    preprocess()
