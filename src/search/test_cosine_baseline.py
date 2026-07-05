"""Testes unitários de cosine_baseline.py. Rodar com: python3 test_cosine_baseline.py"""
from corpus_loader import FALLBACK_SAMPLE_PATH, load_corpus
from cosine_baseline import TfidfCosineSearch


def _fit_on_fixture() -> TfidfCosineSearch:
    # path explícito (não o fallback automático): teste não deve depender de
    # alguém já ter gerado data/processed/movies.parquet nesta máquina.
    corpus = load_corpus(path=FALLBACK_SAMPLE_PATH)
    s = TfidfCosineSearch()
    s.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], tokens=corpus["tokens"])
    return s


def test_search_returns_sorted_scores():
    s = _fit_on_fixture()
    results = s.search("a robot falls in love with a human", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "resultados deveriam vir ordenados por score decrescente"
    assert all(0.0 <= sc <= 1.0001 for sc in scores), "TF-IDF L2-normalizado -> cosseno deveria ficar em [0, 1]"
    print(f"OK: top-1 para 'robot falls in love' é '{results[0].title}' (score={results[0].score:.3f})")


def test_fit_accepts_tokens_kwarg_and_ignores_it():
    """TfidfCosineSearch não usa tokens (o TfidfVectorizer tokeniza sozinho a partir do texto cru),
    mas fit() precisa aceitar o parâmetro pra manter a interface de BaseSemanticSearch."""
    corpus = load_corpus(path=FALLBACK_SAMPLE_PATH)
    s = TfidfCosineSearch()
    s.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"], tokens=corpus["tokens"])
    s_no_tokens = TfidfCosineSearch()
    s_no_tokens.fit(corpus["wiki_movie_id"], corpus["title"], corpus["clean_summary"])  # sem tokens=
    r1 = s.search("murder mystery detective", top_k=3)
    r2 = s_no_tokens.search("murder mystery detective", top_k=3)
    assert [r.wiki_movie_id for r in r1] == [r.wiki_movie_id for r in r2], "tokens não deveria mudar o resultado do TF-IDF"
    print("OK: fit() aceita tokens= (e ignora, como documentado) sem quebrar nem mudar o resultado")


def test_top_k_larger_than_corpus_does_not_crash():
    s = _fit_on_fixture()
    results = s.search("qualquer coisa", top_k=10_000)
    assert len(results) == 45
    print("OK: top_k maior que o corpus não quebra, só retorna todos os documentos")


def test_query_with_no_vocabulary_overlap_returns_zero_scores_not_error():
    s = _fit_on_fixture()
    results = s.search("xyzxyz_termo_totalmente_fora_do_vocabulario", top_k=3)
    assert len(results) == 3
    assert all(r.score == 0.0 for r in results), "sem overlap de vocabulário, score deveria ser 0, não erro"
    print("OK: query sem overlap de vocabulário retorna scores 0.0 em vez de quebrar")


def test_fit_before_search_is_required():
    s = TfidfCosineSearch()
    try:
        s.search("teste")
        raise AssertionError("deveria ter levantado RuntimeError (fit() não foi chamado)")
    except RuntimeError:
        print("OK: search() sem fit() prévio levanta RuntimeError, como esperado")


if __name__ == "__main__":
    test_search_returns_sorted_scores()
    test_fit_accepts_tokens_kwarg_and_ignores_it()
    test_top_k_larger_than_corpus_does_not_crash()
    test_query_with_no_vocabulary_overlap_returns_zero_scores_not_error()
    test_fit_before_search_is_required()
    print("\nTodos os testes de cosine_baseline.py passaram.")
