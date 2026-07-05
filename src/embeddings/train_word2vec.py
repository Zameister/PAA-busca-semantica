import os
import time
from pathlib import Path
import numpy as np
import string

import pandas as pd
from gensim.models import Word2Vec

# ==========================
# Hiperparâmetros Globais
# ==========================

VECTOR_SIZE = 100      # Número de dimensões dos vetores
WINDOW = 5             # Janela de contexto
MIN_COUNT = 1          # Mantém palavras raras
SG = 1                 # Skip-Gram
EPOCHS = 10            # Número de épocas
SEED = 42              # Reprodutibilidade


def main():
    script_start = time.perf_counter()

    # ==========================
    # Caminhos do projeto
    # ==========================

    BASE_DIR = Path(__file__).resolve().parents[2]

    INPUT_PARQUET = BASE_DIR / "data" / "processed" / "movies.parquet"

    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "word2vec_skipgram.model"

    print(" Carregando movies.parquet...")

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{INPUT_PARQUET}\n"
            "Execute o preprocess.py primeiro."
        )

    df = pd.read_parquet(INPUT_PARQUET)

    if "tokens" not in df.columns:
        raise ValueError(
            "A coluna 'tokens' não foi encontrada no DataFrame."
        )

    # ==========================
    # Filtragem das sentenças
    # ==========================

    sentences = [
        [
            token
            for token in (
                tokens.tolist() if isinstance(tokens, np.ndarray) else tokens
            )
            if token not in string.punctuation
        ]
        for tokens in df["tokens"]
        if isinstance(tokens, (list, np.ndarray)) and len(tokens) > 0
    ]

    if not sentences:
        raise ValueError(
            "Nenhuma lista de tokens válida foi encontrada para treinamento."
        )

    # ==========================
    # Métricas do corpus
    # ==========================

    total_docs_original = len(df)
    total_docs_validos = len(sentences)
    total_tokens = sum(len(s) for s in sentences)

    print("\n Corpus")
    print(f"Documentos originais : {total_docs_original:,}")
    print(f"Documentos utilizados: {total_docs_validos:,}")
    print(f"Total de tokens      : {total_tokens:,}")

    # ==========================
    # Configuração do hardware
    # ==========================

    workers = max(1, os.cpu_count() or 1)

    hyperparameters = {
        "vector_size": VECTOR_SIZE,
        "window": WINDOW,
        "min_count": MIN_COUNT,
        "sg": SG,
        "epochs": EPOCHS,
        "workers": workers,
        "seed": SEED,
    }

    print("\n Hiperparâmetros")
    print(f"vector_size : {VECTOR_SIZE}")
    print(f"window      : {WINDOW}")
    print(f"min_count   : {MIN_COUNT}")
    print(f"sg          : {SG}")
    print(f"epochs      : {EPOCHS}")
    print(f"workers     : {workers}")
    print(f"seed        : {SEED}")

    for chave, valor in hyperparameters.items():
        print(f"{chave:12}: {valor}")

    # ==========================
    # Treinamento
    # ==========================

    print("\n Iniciando treinamento...")

    train_start = time.perf_counter()

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        epochs=EPOCHS,
        workers=workers,
        seed=SEED,
    )

    train_end = time.perf_counter()

    # ==========================
    # Informações do modelo
    # ==========================

    print("\n Modelo treinado")
    print(f"Vocabulário           : {len(model.wv):,} palavras")
    print(f"Dimensão dos vetores  : {model.vector_size}")
    print(f"Tempo de treinamento  : {train_end - train_start:.2f} s")

    # ==========================
    # Salvamento
    # ==========================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_start = time.perf_counter()

    model.save(str(MODEL_PATH))

    save_end = time.perf_counter()

    print(f"\n Modelo salvo em:\n{MODEL_PATH}")
    print(f"Tempo de salvamento: {save_end - save_start:.2f} s")

    # ==========================
    # Tempo total
    # ==========================

    script_end = time.perf_counter()

    print("\n Resumo")
    print(f"Treinamento : {train_end - train_start:.2f} s")
    print(f"Salvamento  : {save_end - save_start:.2f} s")
    print(f"Tempo total : {script_end - script_start:.2f} s")


if __name__ == "__main__":
    main()