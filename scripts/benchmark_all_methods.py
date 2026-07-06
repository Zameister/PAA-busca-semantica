"""
benchmark_all_methods.py — benchmark comparativo dos 4 métodos de busca do
projeto, feito na integração final (depois do merge de todas as branches):

    1. TfidfCosineSearch        — Pessoa 2, src/search/cosine_baseline.py
    2. Word2VecAverageSearch    — Pessoa 2, src/search/word2vec_search.py
    3. HeavyRetriever           — Pessoa 3, SBERT + FAISS (src/search/retriever_heavy.py)
    4. HeavyRetriever + rerank  — Pessoa 3, Método 3 + Cross-Encoder (src/search/reranker.py)

Usa as mesmas EXAMPLE_QUERIES do benchmark da Pessoa 2 (src/search/example_queries.py),
pra manter os 4 métodos comparados nas mesmas perguntas.

IMPORTANTE sobre "recall" neste benchmark
------------------------------------------
Recall de verdade precisa de um gabarito de relevância humano (quais filmes
são "corretos" pra cada query) — isso não existe neste repositório, e nem o
benchmark.py da Pessoa 2 tenta calcular isso (o docstring de lá já avisa:
depende de um gabarito que é "tarefa conjunta do grupo").

Não fabricamos esse gabarito à mão. Em vez disso, usamos os GÊNEROS reais de
movies.parquet (dado real do pré-processamento da Pessoa 1, não inventado)
como PROXY objetivo de relevância: um filme é considerado "relevante" pra uma
query se pelo menos um dos seus gêneros bate com o conjunto de gêneros
associado à query (mapeamento abaixo, definido por leitura direta do texto da
query — documentado e revisável pelo grupo).

Isso é só um proxy, não uma medida de relevância de verdade (dois filmes do
gênero "Horror" não são necessariamente igualmente relevantes pra "um monstro
persegue pessoas na floresta à noite"). Fica registrado aqui como limitação
conhecida — uma melhoria futura seria o grupo montar um gabarito de verdade,
mesmo que pequeno (ex: 10-20 filmes julgados manualmente por query).

Como os conjuntos "relevantes por gênero" são grandes (centenas a milhares de
filmes por query) e cada método só retorna top_k=10, os valores de recall@10
são pequenos em valor absoluto pra todos os métodos — isso é esperado, o que
importa pra comparação é a diferença RELATIVA entre os métodos, não o valor
absoluto.
"""

from __future__ import annotations

import sys
import time
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "search"))

from corpus_loader import load_corpus
from cosine_baseline import TfidfCosineSearch
from word2vec_search import Word2VecAverageSearch
from example_queries import EXAMPLE_QUERIES

from src.search.retriever_heavy import HeavyRetriever
from src.search.reranker import rerank_with_cross_encoder

RESULTS_DIR = BASE_DIR / "results"
INDEX_DIR = BASE_DIR / "artifacts" / "heavy_index"
TOP_K = 10
N_LATENCY_REPEATS = 10

# proxy de relevância por gênero (ver docstring acima) -- gêneros conferidos
# contra o vocabulário real de movies.parquet
QUERY_RELEVANT_GENRES = {
    "a detective investigates a murder in the city": {"Crime Fiction", "Mystery", "Detective fiction", "Crime Thriller"},
    "a robot falls in love with a human": {"Science Fiction"},
    "soldiers survive a brutal war": {"War film"},
    "a group plans a heist to steal something valuable": {"Heist"},
    "a monster hunts people in the woods at night": {"Horror", "Monster movie", "Monster"},
}


def _median_latency_ms(fn, repeats=N_LATENCY_REPEATS) -> float:
    fn()  # warmup (carrega modelo etc.) -- não entra na medição
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _recall_at_k(retrieved_ids: list, relevant_ids: set) -> float:
    if not relevant_ids:
        return float("nan")
    hits = len(set(retrieved_ids) & relevant_ids)
    return hits / len(relevant_ids)


def _relevant_ids_by_query() -> dict:
    """load_corpus() (Pessoa 2) não traz a coluna `genres` (fora do escopo da
    busca leve) -- lê movies.parquet direto só pra montar o proxy de relevância."""
    raw = pd.read_parquet(BASE_DIR / "data" / "processed" / "movies.parquet", columns=["wiki_movie_id", "genres"])

    relevant = {}
    for query, genres in QUERY_RELEVANT_GENRES.items():
        mask = raw["genres"].apply(lambda gs: bool(genres & set(gs)))
        relevant[query] = set(raw.loc[mask, "wiki_movie_id"].astype(str))
    return relevant


def benchmark_light_methods(movies_df, relevant_by_query):
    ids, titles, texts, tokens = (
        movies_df["wiki_movie_id"].astype(str), movies_df["title"], movies_df["clean_summary"], movies_df["tokens"],
    )

    rows = []
    for searcher in [TfidfCosineSearch(), Word2VecAverageSearch()]:
        searcher.fit(ids, titles, texts, tokens=tokens)
        for query in EXAMPLE_QUERIES:
            latency = _median_latency_ms(lambda: searcher.search(query, top_k=TOP_K))
            results = searcher.search(query, top_k=TOP_K)
            recall = _recall_at_k([r.wiki_movie_id for r in results], relevant_by_query[query])
            rows.append({"method": searcher.name, "query": query, "latency_ms": latency, "recall_at_10": recall})
    return rows


def benchmark_heavy_methods(relevant_by_query):
    retriever = HeavyRetriever(index_dir=str(INDEX_DIR))

    rows = []
    for query in EXAMPLE_QUERIES:
        latency = _median_latency_ms(lambda: retriever.retrieve(query, top_k=TOP_K))
        results = retriever.retrieve(query, top_k=TOP_K)
        recall = _recall_at_k([str(r["wiki_movie_id"]) for r in results], relevant_by_query[query])
        rows.append({"method": "sbert_faiss", "query": query, "latency_ms": latency, "recall_at_10": recall})

    for query in EXAMPLE_QUERIES:
        def _retrieve_and_rerank():
            candidates = retriever.retrieve(query, top_k=TOP_K * 3)
            return rerank_with_cross_encoder(query, candidates)[:TOP_K]

        latency = _median_latency_ms(_retrieve_and_rerank)
        results = _retrieve_and_rerank()
        recall = _recall_at_k([str(r["wiki_movie_id"]) for r in results], relevant_by_query[query])
        rows.append({"method": "sbert_faiss_rerank", "query": query, "latency_ms": latency, "recall_at_10": recall})

    return rows


def main():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"'{INDEX_DIR}' não encontrado. Gere o índice primeiro com "
            "scripts/precompute_embeddings.py (ver README)."
        )

    movies_df = load_corpus()
    relevant_by_query = _relevant_ids_by_query()

    print("Rodando métodos leves (Pessoa 2)...")
    rows = benchmark_light_methods(movies_df, relevant_by_query)

    print("Rodando métodos pesados (Pessoa 3)...")
    rows += benchmark_heavy_methods(relevant_by_query)

    df = pd.DataFrame(rows)

    summary = df.groupby("method").agg(
        latency_ms_mean=("latency_ms", "mean"),
        recall_at_10_mean=("recall_at_10", "mean"),
    ).reset_index()

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark_all_methods_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "benchmark_all_methods_summary.csv", index=False)

    print("\n" + summary.to_string(index=False))

    _plot_recall_vs_latency(summary)


def _plot_recall_vs_latency(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in summary.iterrows():
        ax.scatter(row["latency_ms_mean"], row["recall_at_10_mean"], s=120)
        ax.annotate(row["method"], (row["latency_ms_mean"], row["recall_at_10_mean"]),
                    textcoords="offset points", xytext=(8, 6))

    ax.set_xlabel("Latência média de busca (ms, escala log)")
    ax.set_ylabel("Recall@10 (proxy por gênero — ver docstring do script)")
    ax.set_xscale("log")
    ax.set_title("Recall vs. latência — comparação dos 4 métodos de busca")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = RESULTS_DIR / "recall_vs_latencia.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nGráfico salvo em {out_path}")


if __name__ == "__main__":
    main()
