# PAA - Busca Semântica em Filmes

Projeto da disciplina de Projeto e Análise de Algoritmos (UnB). Sistema de busca
semântica sobre o CMU Movie Summary Corpus, usando RAG e LLM local.

## Como usar (pra quem não é da parte de pré-processamento)

Os dados brutos e processados (`data/`) não ficam no git — é tudo gerado
localmente. Duas opções:

**Opção 1 — gerar você mesmo (recomendado, é só um comando):**

```bash
python -m venv .venv
.venv\Scripts\activate        # no Windows
pip install -r requirements.txt
python src/preprocessing/preprocess.py
```

O script baixa o dataset, extrai, processa e gera os dois arquivos em
`data/processed/`. Leva alguns minutos na primeira vez (download +
tokenização de ~42 mil resumos); depois disso não precisa rodar de novo.

**Opção 2 — pedir os arquivos prontos:** peça pra pessoa responsável pelo
pré-processamento os arquivos `movies.parquet` e `characters.parquet` (por
Drive, Discord, etc.) e coloque os dois em `data/processed/` — nesse caso
não precisa instalar nada nem rodar o script, só ler os parquets direto
com `pandas.read_parquet(...)`.

Onde encontrar tudo:
- `data/processed/movies.parquet` — um filme por linha (resumo, gêneros,
  tokens etc.)
- `data/processed/characters.parquet` — um personagem/ator por linha
- ambos compartilham a coluna `wiki_movie_id` (veja "Chave de junção" mais
  abaixo) pra quem precisar cruzar os dois

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
6. Lê `character.metadata.tsv` (personagens/atores) e salva em
   `data/processed/characters.parquet`.

**Por que `characters.parquet` é um arquivo separado, e não um join com
`movies.parquet`:** a granularidade é diferente — um filme pode ter dezenas
de personagens, então cada linha de `movies.parquet` viraria várias linhas
repetindo o mesmo resumo e tokens à toa. Quem precisar cruzar os dois pode
fazer o merge por `wiki_movie_id` na hora do uso.

### Como rodar

Ver "Como usar" no topo do README. O download (~1GB descompactado) e a
tokenização de ~42 mil resumos levam alguns minutos na primeira execução.
Nas próximas, o script pula download e extração se os arquivos já
existirem em `data/raw/`.

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

`data/processed/characters.parquet`, com uma linha por personagem/ator:

| coluna                      | descrição                              |
|-----------------------------|------------------------------------------|
| wiki_movie_id                | id do filme na Wikipedia                |
| freebase_movie_id            | id do filme no Freebase                 |
| release_date                 | data de lançamento                      |
| character_name               | nome do personagem                      |
| actor_dob                    | data de nascimento do ator/atriz        |
| actor_gender                 | gênero do ator/atriz                    |
| actor_height                 | altura (m)                              |
| actor_ethnicity               | etnia (id Freebase)                     |
| actor_name                   | nome do ator/atriz                      |
| actor_age_at_release         | idade do ator/atriz no lançamento       |
| freebase_char_actor_map_id   | id Freebase do vínculo personagem-ator  |
| freebase_character_id        | id Freebase do personagem               |
| freebase_actor_id            | id Freebase do ator/atriz               |

### Chave de junção entre os dois arquivos

`movies.parquet` e `characters.parquet` têm granularidades diferentes (um
filme vs. um personagem/ator), mas compartilham a coluna **`wiki_movie_id`**
(id do filme na Wikipedia). Pra cruzar os dois, use ela como chave:

```python
import pandas as pd

movies = pd.read_parquet("data/processed/movies.parquet")
characters = pd.read_parquet("data/processed/characters.parquet")

merged = characters.merge(movies, on="wiki_movie_id", how="left")
```

## Notebook de exploração (notebooks/exploracao.ipynb)

Estatísticas básicas sobre `movies.parquet`: número total de filmes,
distribuição dos gêneros mais comuns e tamanho médio/mediano das sinopses
em número de tokens, com os gráficos correspondentes. Serve como checagem
rápida de sanidade dos dados antes de gerar embeddings.

Pra abrir: instale as dependências do `requirements.txt` (inclui
`ipykernel`) e abra o notebook no VS Code, JupyterLab ou Jupyter Notebook
usando o kernel do `.venv` do projeto.
