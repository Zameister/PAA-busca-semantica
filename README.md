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
3. Salva o resultado em `data/processed/movies.parquet`.
