"""
base_search.py — interface comum para os métodos de busca semântica do projeto.

Definir essa interface agora (com os 2 métodos "leves" da Pessoa 2) facilita a
comparação final entre os 4 métodos: se a Pessoa 3 implementar SBERT+HNSW
seguindo a mesma interface (fit / search), o benchmark.py final consegue tratar
os 4 métodos de forma uniforme, sem código especial pra cada um.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class SearchResult:
    wiki_movie_id: str    # mesmo nome de coluna usado em movies.parquet/characters.parquet
    title: str
    score: float           # similaridade de cosseno, em [-1, 1] (na prática, [0, 1])
    snippet: str            # sinopse truncada, só pra exibição


class BaseSemanticSearch(ABC):
    """Classe-base abstrata: todo método de busca do projeto deve implementar fit() e search()."""

    #: nome curto usado em logs, tabelas de benchmark, etc.
    name: str = "base"

    @abstractmethod
    def fit(
        self,
        movie_ids: Sequence[str],
        titles: Sequence[str],
        texts: Sequence[str],
        tokens: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """
        Constrói o índice de busca a partir do corpus (listas paralelas de
        ids/títulos/textos).

        `tokens`, se fornecido (ex: a coluna `tokens` de movies.parquet, já
        tokenizada via NLTK pela Pessoa 1), é opcional: métodos que trabalham
        em cima de texto puro (TF-IDF) simplesmente ignoram; métodos que
        precisam de tokens (Word2Vec) usam direto em vez de tokenizar `texts`
        de novo à toa.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retorna os top_k filmes mais similares semanticamente à query, ordenados por score desc."""
        raise NotImplementedError

    def _make_snippet(self, text: str, max_chars: int = 160) -> str:
        text = " ".join(text.split())
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
