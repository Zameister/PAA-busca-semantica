"""
benchmark.py — mede tempo e memória de indexação e busca dos métodos da Pessoa 2:
    1) TfidfCosineSearch     (cosine_baseline.py)
    2) Word2VecAverageSearch (word2vec_search.py, requer `pip install gensim`)

Saídas (na pasta results/ da raiz do repo, já existe e é versionada):
    results/benchmark_metrics.csv    — tabela de métricas (tempo/memória de indexação,
                                        latência de busca) pra colar direto nos slides
    results/example_queries.md       — resultados top-k lado a lado dos dois métodos,
                                        pra comparação QUALITATIVA (não só velocidade)

Metodologia de medição (importante documentar no relatório):
    - Tempo: time.perf_counter(), latência de busca = mediana de N repetições por
      query (reduz ruído de cache/SO; média sozinha é sensível a outliers).
    - Memória: delta de RSS (Resident Set Size) do processo via psutil, com
      gc.collect() antes/depois. É uma aproximação — mede o processo inteiro, não
      isola perfeitamente cada método (medições feitas em sequência no mesmo
      processo podem ter efeitos residuais de alocações anteriores). Pra uma
      medição mais rigorosa (mencionem isso como possível extensão no relatório),
      cada método pode ser rodado em um subprocesso isolado.

Este script NÃO calcula recall/precision@K entre os métodos: isso depende de um
gabarito de relevância (quais filmes são "corretos" pra cada query), que é tarefa
conjunta do grupo (ver "Junto no final" na divisão de trabalho), já que só faz
sentido comparar recall depois que os 4 métodos — incluindo SBERT+HNSW da Pessoa 3 —
estiverem prontos.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

# garante que este arquivo importa os módulos vizinhos (base_search, corpus_loader,
# cosine_baseline, example_queries) tanto rodando como script quanto via pytest/import de outro lugar
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base_search import BaseSemanticSearch
from corpus_loader import load_corpus
from cosine_baseline import TfidfCosineSearch
from example_queries import EXAMPLE_QUERIES

# raiz do repo é 2 níveis acima de src/search/benchmark.py — mesma lógica de
# corpus_loader.BASE_DIR e de src/preprocessing/preprocess.py
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
N_QUERY_REPEATS = 20  # repetições por query pra uma latência mediana estável


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _load_word2vec_searcher():
    """Tenta importar/instanciar o método Word2Vec. Retorna None se o gensim não estiver instalado."""
    try:
        import gensim  # noqa: F401
    except ImportError:
        return None
    from word2vec_search import Word2VecAverageSearch
    return Word2VecAverageSearch()


def benchmark_indexing(searcher: BaseSemanticSearch, ids, titles, texts, tokens) -> dict:
    gc.collect()
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    searcher.fit(ids, titles, texts, tokens=tokens)
    elapsed = time.perf_counter() - t0
    gc.collect()
    mem_after = _rss_mb()
    return {
        "index_time_ms": elapsed * 1000,
        "index_memory_mb": max(mem_after - mem_before, 0.0),
    }


def benchmark_search_latency(searcher: BaseSemanticSearch, queries: list[str], repeats: int = N_QUERY_REPEATS) -> dict:
    all_latencies_ms = []
    for q in queries:
        for _ in range(repeats):
            t0 = time.perf_counter()
            searcher.search(q, top_k=5)
            all_latencies_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "search_latency_ms_mean": statistics.mean(all_latencies_ms),
        "search_latency_ms_median": statistics.median(all_latencies_ms),
        "search_latency_ms_stdev": statistics.stdev(all_latencies_ms) if len(all_latencies_ms) > 1 else 0.0,
    }


def run_benchmark(data_path: str | None) -> pd.DataFrame:
    corpus = load_corpus(data_path)
    ids, titles, texts, tokens = (
        corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], corpus["tokens"],
    )

    candidates: list[BaseSemanticSearch] = [TfidfCosineSearch()]
    w2v = _load_word2vec_searcher()
    if w2v is not None:
        candidates.append(w2v)
    else:
        print("[benchmark] AVISO: gensim não instalado — pulando Word2VecAverageSearch. "
              "Rode `pip install gensim` para incluir esse método no benchmark.")

    rows = []
    qualitative_blocks = []

    for searcher in candidates:
        print(f"\n=== {searcher.name} ===")
        idx_metrics = benchmark_indexing(searcher, ids, titles, texts, tokens)
        print(f"  Indexação: {idx_metrics['index_time_ms']:.2f} ms | "
              f"+{idx_metrics['index_memory_mb']:.2f} MB")

        search_metrics = benchmark_search_latency(searcher, EXAMPLE_QUERIES)
        print(f"  Busca (mediana/{N_QUERY_REPEATS}x por query): "
              f"{search_metrics['search_latency_ms_median']:.3f} ms")

        rows.append({
            "method": searcher.name,
            "n_docs": len(corpus),
            **idx_metrics,
            **search_metrics,
        })

        block = [f"## {searcher.name}\n"]
        for q in EXAMPLE_QUERIES:
            results = searcher.search(q, top_k=5)
            block.append(f"**Query:** _{q}_\n")
            if not results:
                block.append("- (sem resultados — todas as palavras da query são OOV)\n")
            for rank, r in enumerate(results, start=1):
                block.append(f"{rank}. `{r.score:.3f}` **{r.title}** — {r.snippet}")
            block.append("")
        qualitative_blocks.append("\n".join(block))

    RESULTS_DIR.mkdir(exist_ok=True)

    metrics_df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[benchmark] Métricas salvas em {metrics_path}")

    queries_path = RESULTS_DIR / "example_queries.md"
    queries_path.write_text(
        "# Comparação qualitativa — resultados top-5 por método\n\n" + "\n\n".join(qualitative_blocks),
        encoding="utf-8",
    )
    print(f"[benchmark] Resultados qualitativos salvos em {queries_path}")

    return metrics_df


def main() -> None:
    # console do Windows usa cp1252 por padrão, que não representa vários
    # caracteres do corpus (ex: "ō"); força utf-8 pra não quebrar o print
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Benchmark dos métodos leves de busca (Pessoa 2)")
    parser.add_argument("--data", type=str, default=None,
                         help="parquet/CSV de entrada (default: data/processed/movies.parquet)")
    args = parser.parse_args()

    metrics_df = run_benchmark(args.data)
    print("\n" + metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
