"""
word2vec_search.py — Método 2 (Pessoa 2): Word2Vec Average + Cosine Similarity brute-force.

Cada sinopse vira UM vetor denso: a média dos vetores Word2Vec das suas palavras
("average word embedding" / bag-of-vectors). A busca continua sendo força bruta
(igual ao Método 1), mas os vetores agora são densos e de dimensão bem menor
(ex: 100) em vez de esparsos e de alta dimensão (tamanho do vocabulário).

Por que treinar o Word2Vec no próprio corpus (em vez de baixar um pré-treinado)?
Treinar do zero no corpus do projeto funciona 100% offline (sem baixar nenhum
modelo externo) e é uma escolha pedagogicamente válida e mais barata em RAM do
que carregar um modelo genérico pré-treinado (que traria vocabulário e domínio
bem diferentes do de sinopses de filme). Se o grupo preferir um pré-treinado
depois, basta trocar a função `_train_word2vec`.

De onde vêm os tokens
----------------------
Quando `fit()` recebe `tokens` (a coluna `tokens` de movies.parquet, já
tokenizada via `nltk.word_tokenize` pela Pessoa 1 em src/preprocessing/preprocess.py),
usamos ela DIRETO — não tokenizamos `texts` de novo à toa. Isso evita repetir
um trabalho O(n·L) que já foi feito uma vez no pré-processamento, e garante que
o vocabulário do Word2Vec bate exatamente com o que o resto do projeto usa.

Se `tokens` não for passado (ex: rodando com um CSV de teste sem essa coluna),
caímos pra tokenizar `texts` aqui mesmo, tentando `nltk.word_tokenize` primeiro
(mesmo tokenizador do pipeline real) e só usando o regex simples do
corpus_loader como último recurso, se os dados do NLTK (punkt) não estiverem
baixados nesta máquina.

Complexidade
------------
Indexação (fit):
    1) Treino do Word2Vec (CBOW, negative sampling): aproximadamente
       O(n · L · w · d), onde n = nº documentos, L = tokens médios por documento,
       w = tamanho da janela de contexto, d = dimensão do embedding.
       É o termo dominante da indexação (bem mais caro que o TF-IDF do Método 1).
    2) Cálculo do vetor médio por documento: O(n · L) — cada palavra é um lookup
       O(1) na tabela de embeddings.

Busca (search), por query:
    1) Tokenizar e calcular vetor médio da query: O(Lq).
    2) Similaridade de cosseno contra os n documentos: O(n · d), estruturalmente
       igual ao Método 1 (força bruta), mas agora d é a dimensão do embedding
       (tipicamente 100–300) em vez do tamanho do vocabulário (pode ser milhares).
       Isso deve tornar a BUSCA mais rápida que o TF-IDF na prática, mesmo com a
       mesma ordem de complexidade assintótica — é o tipo de diferença que só
       aparece medindo (ver benchmark.py), não só olhando o Big-O.
    3) Top-k: O(n) esperado (np.argpartition) + O(k log k).

Limitação conhecida: palavras fora do vocabulário treinado (OOV) são ignoradas na
média. Com um corpus de treino pequeno, isso pode ser uma fração grande da query
— é um ponto justo de comparação contra SBERT (Pessoa 3), que não sofre desse problema.
"""

from __future__ import annotations

import argparse
import time
from typing import Protocol, Sequence

import numpy as np

from base_search import BaseSemanticSearch, SearchResult
from corpus_loader import load_corpus, simple_tokenize
from example_queries import EXAMPLE_QUERIES


class WordVectorsLike(Protocol):
    """
    Interface mínima que precisamos de um modelo de word embeddings (satisfeita por
    gensim's KeyedVectors, ex: Word2Vec(...).wv). Definida pra permitir testar
    average_vector() sem depender do gensim estar instalado.
    """

    vector_size: int

    def __contains__(self, word: str) -> bool: ...
    def __getitem__(self, word: str) -> np.ndarray: ...


def _tokenize(text: str) -> list[str]:
    """
    Tokeniza um texto pro Word2Vec (fit() sem `tokens` fornecido, ou query em
    search()). Tenta nltk.word_tokenize primeiro — o MESMO tokenizador usado em
    src/preprocessing/preprocess.py — pra manter o vocabulário consistente com
    o que gerou a coluna `tokens` de movies.parquet. Se os dados do NLTK (punkt)
    não estiverem baixados nesta máquina, cai pro regex simples do
    corpus_loader (mais pobre — ignora pontuação/contrações — mas nunca quebra).
    """
    text = text.lower()
    try:
        import nltk
        return nltk.word_tokenize(text)
    except (ImportError, LookupError):
        return simple_tokenize(text)


def average_vector(tokens: Sequence[str], wv: WordVectorsLike) -> tuple[np.ndarray, int]:
    """
    Calcula o vetor médio das palavras de `tokens` presentes em `wv`.
    Retorna (vetor, n_palavras_usadas). Palavras fora do vocabulário (OOV) são ignoradas.
    Se nenhuma palavra do texto estiver no vocabulário, retorna vetor de zeros.
    """
    vecs = [wv[t] for t in tokens if t in wv]
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32), 0
    return np.mean(vecs, axis=0).astype(np.float32), len(vecs)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # evita divisão por zero em documentos 100% OOV
    return matrix / norms


class Word2VecAverageSearch(BaseSemanticSearch):
    name = "word2vec_avg"

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 20) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

        self.wv: WordVectorsLike | None = None
        self.doc_matrix: np.ndarray | None = None  # (n_docs x vector_size), L2-normalizada
        self.movie_ids: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []

        # métricas preenchidas em fit()
        self.index_time_seconds: float | None = None
        self.vocab_size: int | None = None
        self.oov_rate: float | None = None  # fração média de palavras OOV por documento (na indexação)

    def _train_word2vec(self, tokenized_docs: list[list[str]]) -> WordVectorsLike:
        """Treina um gensim Word2Vec no próprio corpus. Requer `pip install gensim`."""
        from gensim.models import Word2Vec  # import local: só exige gensim se este método for usado

        model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=0,  # CBOW (mais rápido de treinar que skip-gram; adequado a corpus pequeno)
            epochs=self.epochs,
        )
        return model.wv

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

        if tokens is not None:
            tokenized_docs = [list(t) for t in tokens]
        else:
            tokenized_docs = [_tokenize(t) for t in self.texts]

        self.wv = self._train_word2vec(tokenized_docs)
        self.vocab_size = len(self.wv.key_to_index) if hasattr(self.wv, "key_to_index") else None

        vectors = np.zeros((len(tokenized_docs), self.vector_size), dtype=np.float32)
        oov_fractions = []
        for i, toks in enumerate(tokenized_docs):
            vec, n_used = average_vector(toks, self.wv)
            vectors[i] = vec
            if toks:
                oov_fractions.append(1 - (n_used / len(toks)))

        self.doc_matrix = _l2_normalize_rows(vectors)
        self.oov_rate = float(np.mean(oov_fractions)) if oov_fractions else 0.0
        self.index_time_seconds = time.perf_counter() - t0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.doc_matrix is None or self.wv is None:
            raise RuntimeError("Chame fit() antes de search().")

        query_tokens = _tokenize(query)
        query_vec, n_used = average_vector(query_tokens, self.wv)
        if n_used == 0:
            # nenhuma palavra da query está no vocabulário treinado: não há como rankear por
            # similaridade coerente, então devolvemos lista vazia em vez de um ranking aleatório.
            return []
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)

        scores = self.doc_matrix @ query_vec  # (n_docs,) — já normalizado -> é cosseno direto

        top_k = min(top_k, len(scores))
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
    if not results:
        print("  (nenhuma palavra da query está no vocabulário treinado)")
        return
    for rank, r in enumerate(results, start=1):
        print(f"  {rank}. [{r.score:.3f}] {r.title}  —  {r.snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca semântica: Word2Vec Average (gensim) + Cosine Similarity")
    parser.add_argument("--data", type=str, default=None,
                         help="parquet/CSV de entrada (default: data/processed/movies.parquet)")
    parser.add_argument("--query", type=str, default=None, help="Query única (senão roda queries de exemplo)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-size", type=int, default=100)
    args = parser.parse_args()

    try:
        import gensim  # noqa: F401
    except ImportError:
        raise SystemExit(
            "gensim não está instalado neste ambiente. Rode: pip install gensim\n"
            "(este script depende do gensim para treinar o Word2Vec — ver README.md)"
        )

    corpus = load_corpus(args.data)
    searcher = Word2VecAverageSearch(vector_size=args.vector_size)
    searcher.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], tokens=corpus["tokens"])
    print(f"[word2vec_search] Índice construído em {searcher.index_time_seconds*1000:.2f} ms "
          f"(vocabulário treinado: {searcher.vocab_size} termos, "
          f"OOV médio por doc: {searcher.oov_rate:.1%}, {len(corpus)} documentos)")

    queries = [args.query] if args.query else EXAMPLE_QUERIES
    for q in queries:
        results = searcher.search(q, top_k=args.top_k)
        _print_results(q, results)


if __name__ == "__main__":
    main()
