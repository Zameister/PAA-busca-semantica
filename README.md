# PAA - Busca Semântica em Filmes

Projeto da disciplina de Projeto e Análise de Algoritmos (UnB). Sistema de busca
semântica sobre o CMU Movie Summary Corpus, usando RAG e LLM local.

**Pra estudar cada parte a fundo (o quê, por quê e complexidade de cada
etapa) antes da apresentação, ver [EXPLICACAO_PARTES.md](EXPLICACAO_PARTES.md).**

## Visão geral do sistema

O projeto é dividido em 4 partes, uma por pessoa, formando um pipeline RAG
(Retrieval-Augmented Generation) completo:

```
usuário digita uma pergunta no app Streamlit
        │
        ▼
API FastAPI (src/api/main.py)
        │
        ├─► 1. RETRIEVAL: busca os filmes mais relevantes pra pergunta
        │      (Pessoa 2: TF-IDF / Word2Vec  —  Pessoa 3: SBERT+FAISS + rerank)
        │      entrada: data/processed/movies.parquet (gerado pela Pessoa 1)
        │
        └─► 2. GENERATION: um LLM local (Pessoa 4) recebe a pergunta + os
               filmes encontrados e escreve uma resposta em linguagem natural
        │
        ▼
resposta formatada volta pro app Streamlit, junto com os filmes-fonte
```

Existem 4 métodos de busca implementados (comparados na seção "Benchmark
comparativo" mais abaixo), mas a API usa o método 3
(SBERT+FAISS com reranking) como principal, por ter o melhor trade-off
qualidade/velocidade do grupo.

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

### Instalando as dependências

O projeto usa **requirements separados** por parte, pra ninguém precisar
instalar bibliotecas pesadas (torch, transformers) só pra rodar o
pré-processamento ou a busca leve:

| Arquivo                    | Necessário pra                                          |
|-----------------------------|---------------------------------------------------------|
| `requirements.txt`          | Sempre (pré-processamento, busca leve, notebook)         |
| `requirements-heavy.txt`     | Busca pesada (SBERT+FAISS, reranking) — Pessoa 3         |
| `requirements-llm.txt`       | LLM local, API e app Streamlit — Pessoa 4                |

Pra rodar o sistema completo (API + app), instale os três:

```bash
pip install -r requirements.txt -r requirements-heavy.txt -r requirements-llm.txt
```

### Rodando o sistema completo (API + app)

```bash
# 1. gerar os dados processados (se ainda não tiver, ver "Opção 1" acima)
python src/preprocessing/preprocess.py

# 2. pré-computar o índice de busca pesada (demora alguns minutos, só precisa
#    rodar uma vez -- gera artifacts/heavy_index/, não versionado)
python scripts/precompute_embeddings.py --input data/processed/movies.parquet

# 3. subir a API (numa janela/terminal)
uvicorn src.api.main:app --port 8000

# 4. subir o app (em outra janela/terminal)
streamlit run app/streamlit_app.py
```

O app abre em `http://localhost:8501`. Ele chama a API em
`http://127.0.0.1:8000` — se mudar a porta da API, atualize `API_URL` em
`app/streamlit_app.py`.

## Estrutura

```
data/               dataset baixado e processado (não versionado)
src/preprocessing/  download e pré-processamento dos dados (Pessoa 1)
src/search/         os 4 métodos de busca semântica (Pessoa 2 e Pessoa 3)
src/llm/            formatação de resposta com LLM local (Pessoa 4)
src/api/            API FastAPI que integra busca + LLM (Pessoa 4)
app/                interface Streamlit (Pessoa 4)
scripts/            scripts utilitários (pré-computar índice pesado, benchmark)
notebooks/          notebooks de exploração
artifacts/          índice de busca pesada pré-computado (não versionado)
models/             modelos baixados/treinados (não versionado)
results/            resultados e avaliações (benchmark, notebook)
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

## Busca semântica (src/search/)

4 métodos, dois "leves" (Pessoa 2) e dois "pesados" (Pessoa 3). Todos
recebem o corpus de `movies.parquet` — nenhum reprocessa o texto do zero.
Explicação detalhada de cada um (com complexidade e trade-offs) em
[EXPLICACAO_PARTES.md](EXPLICACAO_PARTES.md).

| Método | Arquivo | Ideia |
|---|---|---|
| 1. TF-IDF + cosseno | `src/search/cosine_baseline.py` | baseline clássico de RI, busca força-bruta sobre vetores esparsos |
| 2. Word2Vec médio + cosseno | `src/search/word2vec_search.py` | embeddings densos treinados no próprio corpus, busca força-bruta |
| 3. SBERT + FAISS | `src/search/retriever_heavy.py` | embeddings de frase (transformer) + índice FAISS `IndexFlatIP` (busca exaustiva/exata, não HNSW aproximado — ver docstring do arquivo) |
| 4. SBERT+FAISS + rerank | `src/search/reranker.py` | pega os candidatos do método 3 e reordena com um Cross-Encoder (mais preciso, mais lento) |

Rodar cada um isoladamente (usa `data/processed/movies.parquet` direto):

```bash
python src/search/cosine_baseline.py
python src/search/word2vec_search.py          # precisa de gensim (requirements.txt)
python src/search/retriever_heavy.py --q "..."  # precisa do índice pré-computado, ver "Como usar"
```

## API + LLM local (src/api/, src/llm/)

`src/api/main.py` expõe `/ask?q=<pergunta>&top_k=<n>`: busca com o método 3
(SBERT+FAISS+rerank), manda os resultados pro LLM local
(`src/llm/answer_formatter.py`, modelo `HuggingFaceTB/SmolLM2-360M-Instruct`,
roda 100% localmente, sem chamada de API externa) formatar uma resposta em
linguagem natural, e devolve resposta + filmes-fonte em JSON.

```bash
uvicorn src.api.main:app --port 8000
curl "http://127.0.0.1:8000/ask?q=um+robo+se+apaixona+por+um+humano&top_k=5"
```

**Limitação conhecida:** o modelo do LLM é pequeno (360M parâmetros, pra
rodar em CPU sem GPU) e às vezes parafraseia demais as sinopses originais em
vez de sintetizar uma resposta totalmente nova — funcional, mas a qualidade
do texto gerado não é perfeita. Ver EXPLICACAO_PARTES.md pra detalhes.

## App (app/streamlit_app.py)

Interface web mínima: campo de busca + slider de quantos filmes considerar,
chama a API e mostra a resposta do LLM junto com os filmes usados como fonte.

```bash
streamlit run app/streamlit_app.py
```

## Benchmark comparativo (scripts/benchmark_all_methods.py)

Compara os 4 métodos nas mesmas 5 perguntas de teste
(`src/search/example_queries.py`), medindo latência de busca e um proxy de
recall (ver limitação abaixo). Gera:

- `results/benchmark_all_methods_raw.csv` — uma linha por (método, query)
- `results/benchmark_all_methods_summary.csv` — médias por método
- `results/recall_vs_latencia.png` — gráfico recall × latência

```bash
python scripts/benchmark_all_methods.py
```

**Resultado (última execução):**

| Método | Latência média | Recall@10 (proxy) |
|---|---|---|
| word2vec_avg | 0.34 ms | 0.0025 |
| cosine_tfidf | 6.34 ms | 0.0046 |
| sbert_faiss | 14.0 ms | 0.0061 |
| sbert_faiss + rerank | 865.5 ms | 0.0065 |

O número do reranking já reflete o modelo carregado uma única vez (ver
"API + LLM local" acima) — os ~865ms são o custo real de rodar o
Cross-Encoder sobre ~30 candidatos, não overhead de recarregar o modelo.

Padrão esperado: métodos mais lentos e sofisticados (SBERT, rerank) acham
mais resultados relevantes; os métodos leves são ordens de magnitude mais
rápidos. Ver o gráfico em `results/recall_vs_latencia.png` pra visualizar
esse trade-off.

**Limitação importante sobre "recall" aqui:** recall de verdade precisa de
um gabarito de relevância humano (quais filmes são "corretos" pra cada
pergunta), que não existe neste repositório. Os números acima usam um PROXY
baseado nos gêneros reais do `movies.parquet` (um filme conta como
"relevante" se o gênero dele bate com o esperado pra pergunta) — é
razoável pra comparar os métodos entre si, mas não é uma medida de
relevância de verdade. Ver o docstring de `benchmark_all_methods.py` pra
detalhes completos. Uma melhoria futura seria o grupo montar um gabarito
de verdade, mesmo que pequeno.

## Notebook de exploração (notebooks/exploracao.ipynb)

Estatísticas básicas sobre `movies.parquet`: número total de filmes,
distribuição dos gêneros mais comuns e tamanho médio/mediano das sinopses
em número de tokens, com os gráficos correspondentes. Serve como checagem
rápida de sanidade dos dados antes de gerar embeddings.

Pra abrir: instale as dependências do `requirements.txt` (inclui
`ipykernel`) e abra o notebook no VS Code, JupyterLab ou Jupyter Notebook
usando o kernel do `.venv` do projeto.
