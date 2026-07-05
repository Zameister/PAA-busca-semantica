"""
cosine_baseline.py — Método 1 (Pessoa 2): Cosine Similarity brute-force sobre TF-IDF.

Este é o baseline clássico de Recuperação de Informação: cada sinopse vira um vetor
esparso de alta dimensão (uma dimensão por termo do vocabulário), pesado por TF-IDF.
A busca é "força bruta": comparamos a query contra TODOS os documentos, sem nenhuma
estrutura de indexação que evite a varredura completa (isso é o que HNSW, da Pessoa 3,
resolve depois).

Complexidade
------------
Indexação (fit):
    Tokenização + contagem de termos: O(n · L), n = nº de documentos, L = tamanho médio
    (em tokens) de cada sinopse.
    Construção da matriz TF-IDF esparsa: O(n · L) também (cada termo de cada doc gera
    no máximo uma entrada não-nula).

Busca (search), por query:
    1) Vetorizar a query: O(Lq) (Lq = nº de tokens da query).
    2) Similaridade de cosseno contra os n documentos: O(n · d) no pior caso, onde
       d = tamanho do vocabulário (dimensão do vetor). Como os vetores TF-IDF do
       sklearn já saem L2-normalizados, cosseno = produto escalar simples, então
       exploramos isso pra evitar recalcular normas a cada busca.
       Na prática, como os vetores são esparsos, o custo real é proporcional ao
       número médio de termos não-nulos por documento (nnz), não a d inteiro.
    3) Seleção dos top-k: O(n) esperado com np.argpartition + O(k log k) pra
       ordenar só os k finais (evita ordenar os n resultados inteiros, que seria
       O(n log n)).

    => Busca dominada por O(n · d) — cresce linearmente com o tamanho do corpus.
       É exatamente esse termo "n" que HNSW ataca, trocando por O(log n) esperado.

Nota: este método NÃO usa a coluna `tokens` do movies.parquet — o TfidfVectorizer
do sklearn faz sua própria tokenização interna a partir do texto cru
(`clean_summary`). O parâmetro `tokens` em fit() existe só pra manter a mesma
assinatura de BaseSemanticSearch; aqui ele é ignorado.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# garante que este arquivo importa os módulos vizinhos (base_search, corpus_loader,
# example_queries) tanto rodando como script quanto via pytest/import de outro lugar
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base_search import BaseSemanticSearch, SearchResult
from corpus_loader import load_corpus
from example_queries import EXAMPLE_QUERIES


class TfidfCosineSearch(BaseSemanticSearch):
    name = "cosine_tfidf"

    def __init__(self, max_features: int | None = 20_000) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",   # troquem pra uma lista PT-BR se o corpus final vier em português
            max_features=max_features,
            norm="l2",              # garante que cosine(u, v) == dot(u, v)
        )
        self.doc_matrix = None   # matriz esparsa (n_docs x vocab_size), L2-normalizada
        self.movie_ids: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []

        # métricas preenchidas em fit()
        self.index_time_seconds: float | None = None
        self.vocab_size: int | None = None

    def fit(
        self,
        movie_ids: Sequence[str],
        titles: Sequence[str],
        texts: Sequence[str],
        tokens: Sequence[Sequence[str]] | None = None,
    ) -> None:
        t0 = time.perf_counter()

        self.movie_ids = list(movie_ids)
        self.titles = list(titles)
        self.texts = list(texts)
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)  # já sai L2-normalizada

        self.index_time_seconds = time.perf_counter() - t0
        self.vocab_size = len(self.vectorizer.vocabulary_)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.doc_matrix is None:
            raise RuntimeError("Chame fit() antes de search().")

        query_vec = self.vectorizer.transform([query])  # (1 x vocab_size), L2-normalizada
        # doc_matrix e query_vec já normalizados -> produto escalar == similaridade de cosseno
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()  # (n_docs,)

        top_k = min(top_k, len(scores))
        # argpartition: O(n) esperado, evita ordenar o vetor inteiro (O(n log n))
        top_idx_unsorted = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx_unsorted[np.argsort(-scores[top_idx_unsorted])]

        return [
            SearchResult(
                wiki_movie_id=self.movie_ids[i],
                title=self.titles[i],
                score=float(scores[i]),
                snippet=self._make_snippet(self.texts[i]),
            )
            for i in top_idx
        ]


def _print_results(query: str, results: list[SearchResult]) -> None:
    print(f'\nQuery: "{query}"')
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. [{r.score:.3f}] {r.title}  —  {r.snippet}")


def main() -> None:
    # console do Windows usa cp1252 por padrão, que não representa vários
    # caracteres do corpus (ex: "ō"); força utf-8 pra não quebrar o print
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Busca semântica: Cosine Similarity + TF-IDF (baseline)")
    parser.add_argument("--data", type=str, default=None,
                         help="parquet/CSV de entrada (default: data/processed/movies.parquet)")
    parser.add_argument("--query", type=str, default=None, help="Query única (senão roda queries de exemplo)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    corpus = load_corpus(args.data)
    searcher = TfidfCosineSearch()
    searcher.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], tokens=corpus["tokens"])
    print(f"[cosine_baseline] Índice construído em {searcher.index_time_seconds*1000:.2f} ms "
          f"(vocabulário: {searcher.vocab_size} termos, {len(corpus)} documentos)")

    queries = [args.query] if args.query else EXAMPLE_QUERIES
    for q in queries:
        results = searcher.search(q, top_k=args.top_k)
        _print_results(q, results)


if __name__ == "__main__":
    main()
