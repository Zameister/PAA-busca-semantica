"""
Testes unitários de corpus_loader.py. Rodar com: python3 test_corpus_loader.py

_adapt_schema() é testada com DataFrames construídos em memória (simulando o
schema real de movies.parquet, incluindo colunas extras e wiki_movie_id como
int) — isso NÃO precisa de pyarrow instalado. A leitura de parquet em si
(_read_raw / pd.read_parquet) não pôde ser testada no ambiente onde este
arquivo foi escrito (sem pyarrow disponível ali) — testem isso de verdade
rodando `python3 corpus_loader.py` depois de gerar data/processed/movies.parquet.
"""
import pandas as pd

from corpus_loader import FALLBACK_SAMPLE_PATH, RAW_REQUIRED_COLUMNS, _adapt_schema, load_corpus

# Schema real completo de movies.parquet (ver README "Saída"), incluindo colunas
# que a busca não usa (genres, runtime, ...) e wiki_movie_id como int — como
# viria de verdade de um TSV lido sem dtype explícito.
_FAKE_RAW_SCHEMA = pd.DataFrame({
    "wiki_movie_id": [975900, 975901, 975902],
    "freebase_movie_id": ["/m/03vyhn", "/m/08yl5d", "/m/0crgdbh"],
    "movie_name": ["Ghosts of Mars", "White Of The Eye", "A Woman in Flames"],
    "release_date": ["2001-08-24", "1987", "1983"],
    "box_office_revenue": [14010832.0, None, None],
    "runtime": [98.0, 111.0, 106.0],
    "languages": [["English Language"], ["English Language"], ["German Language"]],
    "countries": [["United States of America"], ["United Kingdom"], ["Germany"]],
    "genres": [["Science Fiction", "Horror"], ["Thriller"], ["Drama"]],
    "summary": ["raw <b>html</b> summary...", "raw summary two...", None],  # None de propósito
    "clean_summary": [
        "set in the second half of the 22nd century...",
        "a police officer investigates a series of murders...",
        "a woman leaves her husband to work in a brothel...",
    ],
    "tokens": [
        ["set", "in", "the", "second", "half", "of", "the", "22nd", "century"],
        ["a", "police", "officer", "investigates", "a", "series", "of", "murders"],
        ["a", "woman", "leaves", "her", "husband"],
    ],
})


def test_adapt_schema_maps_real_column_names():
    result = _adapt_schema(_FAKE_RAW_SCHEMA, source_label="teste")
    assert list(result.columns) == ["wiki_movie_id", "title", "clean_summary", "tokens"], \
        "não deveria vazar genres/runtime/etc — só o que a busca precisa"
    assert result["title"].tolist() == ["Ghosts of Mars", "White Of The Eye", "A Woman in Flames"]
    print("OK: _adapt_schema mapeia wiki_movie_id/movie_name/clean_summary corretamente")


def test_adapt_schema_converts_id_to_string():
    result = _adapt_schema(_FAKE_RAW_SCHEMA, source_label="teste")
    # não checamos dtype == object: no pandas 3.x (pinado no requirements.txt)
    # o dtype padrão de string mudou, mas os VALORES continuam sendo str — é
    # isso que importa pra concatenação/comparação/exibição no resto do código.
    assert all(isinstance(v, str) for v in result["wiki_movie_id"])
    assert result["wiki_movie_id"].tolist() == ["975900", "975901", "975902"]
    print("OK: wiki_movie_id (int no parquet real) vira string")


def test_adapt_schema_uses_precomputed_tokens_as_is():
    result = _adapt_schema(_FAKE_RAW_SCHEMA, source_label="teste")
    assert result["tokens"].iloc[0] == ["set", "in", "the", "second", "half", "of", "the", "22nd", "century"]
    assert all(isinstance(t, list) for t in result["tokens"])
    print("OK: tokens pré-computados (NLTK) são usados como estão, sem re-tokenizar")


def test_adapt_schema_falls_back_to_simple_tokenize_without_tokens_column():
    raw_no_tokens = _FAKE_RAW_SCHEMA.drop(columns=["tokens"])
    result = _adapt_schema(raw_no_tokens, source_label="teste")
    assert result["tokens"].iloc[1] == ["a", "police", "officer", "investigates", "a", "series", "of", "murders"]
    print("OK: sem coluna `tokens`, cai pro tokenizador regex de fallback (mesmo resultado aqui, por coincidência)")


def test_adapt_schema_drops_rows_with_missing_summary_and_keeps_alignment():
    result = _adapt_schema(_FAKE_RAW_SCHEMA, source_label="teste")
    # a 3a linha tem summary=None (não clean_summary=None) -- não deveria ser afetada.
    # mas se clean_summary fosse None, a linha deveria sumir E tokens não deveria desalinhar:
    raw_with_missing_clean = _FAKE_RAW_SCHEMA.copy()
    raw_with_missing_clean.loc[1, "clean_summary"] = None
    result2 = _adapt_schema(raw_with_missing_clean, source_label="teste")
    assert len(result2) == 2, "linha com clean_summary=None deveria ser descartada"
    assert result2["title"].tolist() == ["Ghosts of Mars", "A Woman in Flames"]
    # o ponto crítico: tokens da linha remanescente batem com o filme certo (não desalinharam)
    assert result2.loc[result2["title"] == "A Woman in Flames", "tokens"].iloc[0] == \
        ["a", "woman", "leaves", "her", "husband"]
    print("OK: linhas com clean_summary ausente são descartadas sem desalinhar tokens/título")


def test_adapt_schema_raises_on_missing_required_column():
    try:
        _adapt_schema(_FAKE_RAW_SCHEMA.drop(columns=["movie_name"]), source_label="teste")
        raise AssertionError("deveria ter levantado ValueError")
    except ValueError as e:
        assert "movie_name" in str(e)
        print("OK: falta de coluna obrigatória levanta ValueError claro")


def test_load_corpus_end_to_end_with_csv_fixture():
    """Único teste que toca disco de verdade — usa o CSV (não precisa de pyarrow)."""
    corpus = load_corpus(path=FALLBACK_SAMPLE_PATH)
    assert len(corpus) == 45
    assert list(corpus.columns) == ["wiki_movie_id", "title", "clean_summary", "tokens"]
    assert all(isinstance(t, list) for t in corpus["tokens"])
    print(f"OK: load_corpus(path=FALLBACK_SAMPLE_PATH) end-to-end funciona ({len(corpus)} filmes)")


if __name__ == "__main__":
    test_adapt_schema_maps_real_column_names()
    test_adapt_schema_converts_id_to_string()
    test_adapt_schema_uses_precomputed_tokens_as_is()
    test_adapt_schema_falls_back_to_simple_tokenize_without_tokens_column()
    test_adapt_schema_drops_rows_with_missing_summary_and_keeps_alignment()
    test_adapt_schema_raises_on_missing_required_column()
    test_load_corpus_end_to_end_with_csv_fixture()
    print("\nTodos os testes de corpus_loader.py passaram.")
