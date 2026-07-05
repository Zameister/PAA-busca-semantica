"""
answer_formatter.py — Pessoa 4: usa um LLM local (sem chamadas de API externa)
pra transformar uma lista de filmes recuperados pela busca (Pessoa 2/3) numa
resposta em linguagem natural pro usuário. Esta é a etapa de "geração" do RAG;
a etapa de "recuperação" (retrieval) é responsabilidade de src/search/.

Modelo: HuggingFaceTB/SmolLM2-360M-Instruct — pequeno o bastante pra rodar em
CPU em tempo razoável (não precisa de GPU), mas ainda segue instruções básicas
de formatação. Baixado uma vez do Hugging Face Hub e cacheado localmente; a
inferência em si não depende de rede nem de nenhuma API paga.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SYSTEM_PROMPT = (
    "Você é um assistente que recomenda filmes de forma breve e natural, em "
    "português, com base apenas nos filmes encontrados pela busca. Não invente "
    "filmes que não estejam na lista."
)

_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
_pipeline = None  # carregado sob demanda (lazy) e reusado entre chamadas


@dataclass
class MovieHit:
    """Formato mínimo esperado de um resultado de busca (ver SearchResult em
    src/search/base_search.py, da Pessoa 2 — mesmos campos, definidos aqui de
    novo pra este módulo não depender de src/search/ ainda não estar mesclado)."""
    title: str
    snippet: str
    score: float = 0.0


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-generation", model=_MODEL_NAME)
    return _pipeline


def _build_messages(query: str, movies: Sequence[MovieHit]) -> list[dict]:
    movies_block = "\n".join(
        f"{i}. {m.title} — {m.snippet}" for i, m in enumerate(movies, start=1)
    )
    user_prompt = (
        f"Pergunta do usuário: {query}\n"
        f"Filmes encontrados pela busca:\n{movies_block}\n\n"
        "Escreva uma resposta curta (2-4 frases) recomendando esses filmes pro "
        "usuário, explicando rapidamente por que combinam com a pergunta."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def format_answer(query: str, movies: Sequence[MovieHit], max_new_tokens: int = 150) -> str:
    """Gera a resposta final em linguagem natural a partir da query + filmes recuperados.

    Complexidade: O(1) chamada ao modelo por resposta (não é O(n) no tamanho do
    corpus — o LLM só vê os poucos filmes já filtrados pela busca, não o corpus
    inteiro). O custo real é o de inferência do LLM em si (proporcional a
    max_new_tokens), independente de n.
    """
    if not movies:
        return "Não encontrei nenhum filme que combine com essa busca."

    pipe = _get_pipeline()
    messages = _build_messages(query, movies)
    output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()


if __name__ == "__main__":
    example_movies = [
        MovieHit(title="Robosapien: Rebooted", snippet="um robô chamado Cody faz amizade com um garoto, e a mãe do garoto se apaixona pelo inventor.", score=0.62),
        MovieHit(title="Miss Susie Slagle's", snippet="uma estudante de enfermagem se apaixona por um interno.", score=0.58),
    ]
    print(format_answer("um robô se apaixona por um humano", example_movies))
