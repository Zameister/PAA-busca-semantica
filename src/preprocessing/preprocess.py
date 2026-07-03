"""
Pre-processamento do CMU Movie Summary Corpus.

Baixa o dataset, junta os resumos de enredo com os metadados dos filmes,
limpa e tokeniza o texto, e salva tudo em data/processed/movies.parquet.
"""

import ast
import re
import tarfile
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup

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

CHARACTER_COLUMNS = [
    "wiki_movie_id",
    "freebase_movie_id",
    "release_date",
    "character_name",
    "actor_dob",
    "actor_gender",
    "actor_height",
    "actor_ethnicity",
    "actor_name",
    "actor_age_at_release",
    "freebase_char_actor_map_id",
    "freebase_character_id",
    "freebase_actor_id",
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


def ensure_nltk_resources():
    resources = {
        "punkt_tab": "tokenizers/punkt_tab",
        "punkt": "tokenizers/punkt",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


def load_plot_summaries():
    path = EXTRACT_DIR / "plot_summaries.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["wiki_movie_id", "summary"])
    return df


def load_metadata():
    path = EXTRACT_DIR / "movie.metadata.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=METADATA_COLUMNS)
    return df


def load_characters():
    path = EXTRACT_DIR / "character.metadata.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=CHARACTER_COLUMNS)
    return df


def extract_names(freebase_dict_str):
    """Campos de gênero/idioma/país vêm como string de dict {freebase_id: nome}."""
    try:
        d = ast.literal_eval(freebase_dict_str)
        return [strip_surrogates(v) for v in d.values()]
    except (ValueError, SyntaxError):
        return []


def strip_surrogates(text):
    """O corpus tem alguns bytes mal decodificados que viram surrogates inválidos em utf-8."""
    return text.encode("utf-8", "ignore").decode("utf-8")


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = strip_surrogates(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess():
    download_dataset()
    extract_dataset()
    ensure_nltk_resources()

    summaries = load_plot_summaries()
    metadata = load_metadata()

    df = summaries.merge(metadata, on="wiki_movie_id", how="inner")

    df["genres"] = df["genres"].apply(extract_names)
    df["languages"] = df["languages"].apply(extract_names)
    df["countries"] = df["countries"].apply(extract_names)

    df["clean_summary"] = df["summary"].apply(clean_text)
    df["tokens"] = df["clean_summary"].apply(nltk.word_tokenize)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "movies.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{len(df)} filmes processados -> {out_path}")
    return df


def preprocess_characters():
    """
    Salvo em arquivo separado de movies.parquet porque a granularidade é
    diferente: um filme tem várias linhas aqui (uma por personagem/ator),
    então juntar com movies.parquet duplicaria resumo e tokens várias vezes
    por filme à toa.
    """
    download_dataset()
    extract_dataset()

    df = load_characters()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "characters.parquet"
    df.to_parquet(out_path, index=False)

    print(f"{len(df)} personagens processados -> {out_path}")
    return df


if __name__ == "__main__":
    preprocess()
    preprocess_characters()
