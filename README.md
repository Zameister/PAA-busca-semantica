# PAA - Busca Semântica em Filmes

Projeto da disciplina de Projeto e Análise de Algoritmos (UnB). Sistema de busca
semântica sobre o CMU Movie Summary Corpus, usando RAG e LLM local.

## Estrutura

```
data/               dataset baixado e processado (não versionado)
src/preprocessing/  download e pré-processamento dos dados
src/search/         busca semântica / embeddings
src/llm/            LLM local
src/api/            API (FastAPI)
app/                interface (Streamlit)
notebooks/          notebooks de exploração
models/             modelos baixados/treinados (não versionado)
results/            resultados e avaliações
```

## Pré-processamento (src/preprocessing/preprocess.py)

O que o script faz:

1. Baixa o [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz)
   e extrai em `data/raw/`.
2. Lê `plot_summaries.txt` (resumos) e `movie.metadata.tsv` (metadados) e junta
   os dois pelo id do filme.
3. Limpa o texto dos resumos: remove HTML, normaliza unicode, deixa em minúsculo.
4. Tokeniza os resumos limpos com NLTK.
5. Salva tudo em `data/processed/movies.parquet`.

### Saída

`data/processed/movies.parquet`, com uma linha por filme e as colunas:

| coluna              | descrição                                          |
|---------------------|-----------------------------------------------------|
| wiki_movie_id        | id do filme na Wikipedia                            |
| summary              | resumo original                                     |
| movie_name           | título do filme                                     |
| release_date         | data de lançamento                                  |
| box_office_revenue   | bilheteria                                           |
| runtime              | duração (min)                                        |
| languages            | lista de idiomas                                    |
| countries            | lista de países                                     |
| genres               | lista de gêneros                                    |
| clean_summary        | resumo limpo (sem HTML, minúsculo, normalizado)     |
| tokens               | lista de tokens do `clean_summary` (NLTK)           |

Esse parquet é a entrada esperada pelo módulo de busca semântica
(`src/search/`) para gerar os embeddings.
