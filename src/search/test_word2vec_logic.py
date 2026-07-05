"""
Testes unitários de word2vec_search.py.

Usam um KeyedVectors FALSO (embeddings aleatórios determinísticos) no lugar do
gensim real, então validam toda a lógica ao redor do treino — average pooling,
normalização L2, produto escalar, top-k, tratamento de OOV, reuso de tokens
pré-computados — sem precisar treinar um Word2Vec de verdade nem ter o gensim
instalado. Isso NÃO testa se o gensim.Word2Vec em si funciona (é uma biblioteca
de terceiros, não é nosso código), só garante que o código escrito por nós ao
redor dele está correto. Rodar com: python3 test_word2vec_logic.py
"""
from unittest.mock import patch

import numpy as np

from corpus_loader import FALLBACK_SAMPLE_PATH, load_corpus
from word2vec_search import Word2VecAverageSearch, _tokenize, average_vector, _l2_normalize_rows


class FakeKeyedVectors:
    """Substitui gensim's Word2Vec(...).wv: embeddings determinísticos por hash da palavra."""

    def __init__(self, vocab: set[str], vector_size: int = 100, seed: int = 42):
        self.vector_size = vector_size
        rng = np.random.default_rng(seed)
        self._table = {w: rng.normal(size=vector_size).astype(np.float32) for w in vocab}
        self.key_to_index = {w: i for i, w in enumerate(self._table)}

    def __contains__(self, word):
        return word in self._table

    def __getitem__(self, word):
        return self._table[word]


def test_average_vector_basic():
    wv = FakeKeyedVectors(vocab={"a", "b", "c"}, vector_size=4)
    vec, n_used = average_vector(["a", "b", "zzz_oov"], wv)
    expected = (wv["a"] + wv["b"]) / 2
    assert n_used == 2, f"esperado 2 palavras usadas, veio {n_used}"
    assert np.allclose(vec, expected), "média não bate com o esperado"
    print("OK: average_vector calcula a média correta e ignora OOV")


def test_average_vector_all_oov():
    wv = FakeKeyedVectors(vocab={"a", "b"}, vector_size=4)
    vec, n_used = average_vector(["zzz", "yyy"], wv)
    assert n_used == 0
    assert np.allclose(vec, np.zeros(4)), "esperado vetor de zeros quando tudo é OOV"
    print("OK: average_vector devolve zeros quando 100% das palavras são OOV")


def test_l2_normalize_rows():
    m = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    normed = _l2_normalize_rows(m.copy())
    assert np.allclose(np.linalg.norm(normed[0]), 1.0)
    assert np.allclose(normed[1], [0.0, 0.0]), "linha de zeros não deveria virar NaN"
    assert np.allclose(np.linalg.norm(normed[2]), 1.0)
    print("OK: _l2_normalize_rows normaliza e trata vetores nulos sem gerar NaN/divisão por zero")


def test_tokenize_falls_back_to_regex_without_nltk_data():
    """
    nltk é dependência obrigatória do projeto (requirements.txt), então não dá
    pra testar o fallback contando com ele estar ausente do ambiente. Em vez
    disso, forçamos o LookupError que nltk.word_tokenize levantaria se os
    dados do punkt não estivessem baixados, pra testar o fallback de forma
    determinística em qualquer máquina.
    """
    import nltk

    with patch.object(nltk, "word_tokenize", side_effect=LookupError("punkt não baixado (simulado)")):
        tokens = _tokenize("Hello, World! It's a test.")

    assert tokens == ["hello", "world", "it", "s", "a", "test"], tokens
    print("OK: _tokenize cai pro regex de fallback quando os dados do nltk não estão disponíveis "
          f"(resultado: {tokens})")


def test_fit_uses_precomputed_tokens_directly_without_retokenizing():
    """
    Ponto central da adaptação ao projeto real: fit() deve usar os tokens
    prontos (vindos de movies.parquet, tokenizados via NLTK pela Pessoa 1) em
    vez de tokenizar `texts` de novo. Aqui provamos isso passando tokens
    propositalmente DIFERENTES do texto real, e conferindo que o vocabulário
    treinado reflete os tokens passados, não o texto.
    """
    class TestableSearch(Word2VecAverageSearch):
        def _train_word2vec(self, tokenized_docs):
            vocab = {t for doc in tokenized_docs for t in doc}
            return FakeKeyedVectors(vocab, vector_size=self.vector_size)

    ids = ["1", "2"]
    titles = ["Filme 1", "Filme 2"]
    texts = ["este texto não importa pro teste", "nem este aqui"]
    fake_tokens = [["palavra_marcada_unica_a"], ["palavra_marcada_unica_b"]]

    s = TestableSearch(vector_size=8)
    s.fit(ids, titles, texts, tokens=fake_tokens)

    assert "palavra_marcada_unica_a" in s.wv.key_to_index
    assert "este" not in s.wv.key_to_index, (
        "fit() deveria ter usado os tokens fornecidos, não re-tokenizado `texts`"
    )
    print("OK: fit() usa os tokens pré-computados fornecidos, sem re-tokenizar texts internamente")


def test_fit_and_search_end_to_end_with_fake_embeddings():
    """
    Testa fit()/search() da classe real Word2VecAverageSearch, mas substituindo só o
    treino do gensim (_train_word2vec) por embeddings falsos determinísticos.
    Isso valida TODO o resto do pipeline (average pooling, normalização, produto
    escalar, argpartition/top-k, construção de SearchResult) sem precisar do gensim.
    """
    corpus = load_corpus(path=FALLBACK_SAMPLE_PATH)

    class TestableSearch(Word2VecAverageSearch):
        def _train_word2vec(self, tokenized_docs):
            vocab = {t for doc in tokenized_docs for t in doc}
            return FakeKeyedVectors(vocab, vector_size=self.vector_size)

    searcher = TestableSearch(vector_size=32)
    searcher.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], tokens=corpus["tokens"])

    assert searcher.doc_matrix.shape == (len(corpus), 32)
    norms = np.linalg.norm(searcher.doc_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"normas fora do esperado: {norms.min()=} {norms.max()=}"

    results = searcher.search("robot love human", top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "resultados deveriam vir ordenados por score decrescente"
    assert all(-1.0001 <= s <= 1.0001 for s in scores), "score de cosseno fora do intervalo [-1, 1]"

    results_all = searcher.search("robot", top_k=999)
    assert len(results_all) == len(corpus)

    results_oov = searcher.search("zzz yyy xxx_inexistente", top_k=3)
    assert results_oov == []

    print(f"OK: fit()/search() ponta-a-ponta funcionam (testado com embeddings falsos, "
          f"{len(corpus)} docs, top-3 exemplo: {[r.title for r in results]})")


if __name__ == "__main__":
    test_average_vector_basic()
    test_average_vector_all_oov()
    test_l2_normalize_rows()
    test_tokenize_falls_back_to_regex_without_nltk_data()
    test_fit_uses_precomputed_tokens_directly_without_retokenizing()
    test_fit_and_search_end_to_end_with_fake_embeddings()
    print("\nTodos os testes de lógica passaram (sem depender do gensim estar instalado).")
