import time
import string
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def calculate_average_vector(tokens, model):
    """
    Calcula a média dos vetores das palavras do Word2Vec,
    removendo pontuação e tokens fora do vocabulário.
    """

    if tokens is None:
        return np.zeros(model.vector_size)

    # Normaliza tipo (Parquet pode retornar list ou numpy.ndarray)
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    if not isinstance(tokens, list) or len(tokens) == 0:
        return np.zeros(model.vector_size)

    # Remove pontuação
    valid_tokens = [
        token
        for token in tokens
        if token not in string.punctuation
    ]

    # Filtra palavras conhecidas pelo modelo
    valid_vectors = [
        model.wv[token]
        for token in valid_tokens
        if token in model.wv
    ]

    if len(valid_vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(valid_vectors, axis=0)


def main():
    script_start = time.perf_counter()

    # ==========================
    # Caminhos do projeto
    # ==========================
    BASE_DIR = Path(__file__).resolve().parents[2]

    INPUT_PARQUET = BASE_DIR / "data" / "processed" / "movies.parquet"
    MODEL_PATH = BASE_DIR / "models" / "word2vec_skipgram.model"

    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    EMBEDDINGS_OUTPUT = OUTPUT_DIR / "word2vec_embeddings.npy"
    METADATA_OUTPUT = OUTPUT_DIR / "movies_metadata.parquet"

    print("Carregando dataset e modelo...")

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {INPUT_PARQUET}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    df = pd.read_parquet(INPUT_PARQUET)
    model = Word2Vec.load(str(MODEL_PATH))

    if "tokens" not in df.columns:
        raise ValueError("Coluna 'tokens' não encontrada no dataset.")

    # ==========================
    # Cálculo dos embeddings
    # ==========================
    print("\n Calculando Word2Vec Average...")

    process_start = time.perf_counter()

    embeddings_list = [
        calculate_average_vector(tokens, model)
        for tokens in df["tokens"]
    ]

    embeddings_matrix = np.vstack(embeddings_list)

    metadata_df = df[["wiki_movie_id", "movie_name"]].copy()

    metadata_df["is_empty_embedding"] = [
        np.linalg.norm(v) == 0
        for v in embeddings_list
    ]

    process_end = time.perf_counter()

    # ==========================
    # Metadados
    # ==========================
    metadata_df = df[["wiki_movie_id", "movie_name"]].copy()

    # ✔ Correção importante: verifica vetor nulo corretamente
    metadata_df["is_empty_embedding"] = [
        np.linalg.norm(x) == 0 for x in embeddings_list
    ]

    # ==========================
    # Salvamento
    # ==========================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n Salvando embeddings em:\n{EMBEDDINGS_OUTPUT}")
    save_start = time.perf_counter()
    np.save(EMBEDDINGS_OUTPUT, embeddings_matrix)
    save_end = time.perf_counter()

    print(f" Salvando metadados em:\n{METADATA_OUTPUT}")
    metadata_df.to_parquet(METADATA_OUTPUT, index=False)

    print(f"Tempo de salvamento : {save_end-save_start:.2f}s")

    # ==========================
    # Estatísticas finais
    # ==========================
    script_end = time.perf_counter()

    total_filmes = embeddings_matrix.shape[0]
    tempo_calculo = process_end - process_start

    print("\n Resumo")
    print(f"Dimensão do modelo    : {model.vector_size}")
    print(f"Embeddings shape      : {embeddings_matrix.shape}")
    print(f"Filmes sem embedding  : {metadata_df['is_empty_embedding'].sum():,}")
    print(f"Tempo cálculo         : {tempo_calculo:.2f} s")
    print(f"Tempo médio por filme : {(tempo_calculo / total_filmes) * 1000:.4f} ms")
    print(f"Tempo total script    : {script_end - script_start:.2f} s")


if __name__ == "__main__":
    main()