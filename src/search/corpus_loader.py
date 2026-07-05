"""
corpus_loader.py — ponto único de carregamento do corpus para os métodos de
busca da Pessoa 2 (src/search/).

Fonte primária esperada: data/processed/movies.parquet, gerado por
src/preprocessing/preprocess.py (Pessoa 1). Ver README.md ("Saída") pro
esquema completo — este loader usa só 4 das colunas desse parquet:

    wiki_movie_id   -> id do filme (mesma chave usada pra cruzar com
                       characters.parquet, ver README "Chave de junção")
    movie_name      -> título
    clean_summary   -> sinopse limpa (entrada do TF-IDF e do Word2Vec)
    tokens          -> lista de tokens NLTK do clean_summary, já pronta —
                       o Word2Vec usa direto em vez de tokenizar de novo à toa

Se data/processed/movies.parquet ainda não existir (ninguém rodou o
pré-processamento ainda nesta máquina), caímos para uma fixture sintética
pequena em fixtures/sample_movies.csv — só pra dev/teste, ver aviso no
próprio README sobre isso.

Nota: a leitura de parquet (via pd.read_parquet, motor pyarrow) não pôde
ser testada no ambiente onde este arquivo foi escrito — não havia pyarrow
disponível ali. A função foi separada em duas partes (_read_raw / _adapt_schema)
exatamente por causa disso: _adapt_schema é testável com um DataFrame em
memória, sem precisar de um parquet de verdade — ver test_corpus_loader.py.
Testem a leitura do parquet real de vocês antes de considerar isso 100% ok.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# mesma lógica de src/preprocessing/preprocess.py: 2 níveis acima deste
# arquivo (src/search/corpus_loader.py) é a raiz do repo.
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET_PATH = BASE_DIR / "data" / "processed" / "movies.parquet"
FALLBACK_SAMPLE_PATH = Path(__file__).parent / "fixtures" / "sample_movies.csv"

# colunas cruas exigidas do preprocess.py da Pessoa 1 (ou de um CSV compatível)
RAW_REQUIRED_COLUMNS = {"wiki_movie_id", "movie_name", "clean_summary"}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenizador de fallback (regex, lowercase) — usado só quando a fonte não
    traz uma coluna `tokens` pronta (ex: a fixture CSV de teste). O pipeline
    real (Pessoa 1) já entrega tokens via NLTK no parquet; isto NÃO tenta
    imitar o NLTK, é só uma rede de segurança pra nunca quebrar por falta de
    tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def _ensure_token_list(value) -> list[str]:
    """Normaliza a célula de `tokens` (list, np.ndarray ou NaN) pra list[str]."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    return list(value)


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Extensão não suportada: '{path.suffix}' (esperado .parquet ou .csv)")


def _adapt_schema(df_raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Recebe um DataFrame com as colunas cruas do preprocess.py da Pessoa 1 (ou
    um CSV compatível) e devolve só o que os métodos de busca precisam:
    wiki_movie_id, title, clean_summary, tokens (list[str]).

    De propósito sem NENHUM I/O aqui — só transformação. Isso permite testar
    esta função inteira com um DataFrame construído em memória, sem depender
    de conseguir ler um parquet/CSV real em disco.
    """
    missing = RAW_REQUIRED_COLUMNS - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"'{source_label}' não tem as colunas obrigatórias: {sorted(missing)}. "
            f"Colunas encontradas: {list(df_raw.columns)}"
        )

    # dropna + reset ANTES de fatiar colunas, pra tokens/summary não desalinharem
    df_raw = df_raw.dropna(subset=["clean_summary"]).reset_index(drop=True)

    result = pd.DataFrame({
        "wiki_movie_id": df_raw["wiki_movie_id"].astype(str),
        "title": df_raw["movie_name"],
        "clean_summary": df_raw["clean_summary"],
    })

    if "tokens" in df_raw.columns:
        result["tokens"] = df_raw["tokens"].apply(_ensure_token_list)
    else:
        result["tokens"] = result["clean_summary"].apply(simple_tokenize)

    return result


def load_corpus(path: str | Path | None = None) -> pd.DataFrame:
    """
    Carrega o corpus e devolve um DataFrame com colunas:
        wiki_movie_id, title, clean_summary, tokens (list[str])

    Parameters
    ----------
    path : caminho explícito pro parquet/CSV (ex: vindo de --data). Se None,
           tenta data/processed/movies.parquet (saída real do preprocess.py)
           e, se não existir, cai pra fixtures/sample_movies.csv (sintética).
    """
    used_fallback = False

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus não encontrado em '{path}'.")
    else:
        path = DEFAULT_PARQUET_PATH
        if not path.exists():
            if not FALLBACK_SAMPLE_PATH.exists():
                raise FileNotFoundError(
                    f"'{DEFAULT_PARQUET_PATH}' não existe (rode "
                    f"src/preprocessing/preprocess.py primeiro, ou peça os "
                    f"parquets prontos — ver README) e a fixture de teste "
                    f"'{FALLBACK_SAMPLE_PATH}' também não foi encontrada."
                )
            print(
                f"[corpus_loader] AVISO: '{DEFAULT_PARQUET_PATH}' não encontrado. "
                f"Rode src/preprocessing/preprocess.py pra gerá-lo (ou peça os "
                f"parquets prontos pra Pessoa 1 — ver README). Usando fixture "
                f"sintética em '{FALLBACK_SAMPLE_PATH}' por enquanto.",
                file=sys.stderr,
            )
            path = FALLBACK_SAMPLE_PATH
            used_fallback = True

    df_raw = _read_raw(path)
    df = _adapt_schema(df_raw, source_label=str(path))

    label = "fixture sintética" if used_fallback else f"'{path}'"
    print(f"[corpus_loader] {len(df)} filmes carregados de {label}.", file=sys.stderr)

    return df


if __name__ == "__main__":
    corpus = load_corpus()
    print(corpus[["wiki_movie_id", "title"]].head())
    print("Exemplo de tokens:", corpus["tokens"].iloc[0][:10])
