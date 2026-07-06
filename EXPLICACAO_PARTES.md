# Explicação de cada parte do projeto

Este documento é material de estudo, não um resumo técnico pra registro. A
ideia é que cada pessoa leia a própria seção (e dê uma olhada nas outras)
antes da apresentação e consiga explicar, com as próprias palavras, o que o
código faz, por que essa abordagem foi escolhida, e o que a complexidade dele
significa na prática — sem precisar decorar nada.

Índice:
- [Pessoa 1 — Pré-processamento dos dados](#pessoa-1--pré-processamento-dos-dados)
- [Pessoa 2 — Busca leve (TF-IDF e Word2Vec)](#pessoa-2--busca-leve-tf-idf-e-word2vec)
- [Pessoa 3 — Busca pesada (SBERT + FAISS + Reranking)](#pessoa-3--busca-pesada-sbert--faiss--reranking)
- [Pessoa 4 — LLM local, API e App](#pessoa-4--llm-local-api-e-app)
- [Visão geral do fluxo completo](#visão-geral-do-fluxo-completo)

---

## Pessoa 1 — Pré-processamento dos dados

**Código:** `src/preprocessing/preprocess.py`

### O que o código faz, passo a passo

1. Baixa o CMU Movie Summary Corpus (um `.tar.gz` de ~50MB) direto do site da
   pesquisa, e extrai os arquivos numa pasta local.
2. Lê dois arquivos de texto tabulado (TSV/CSV com tab): `plot_summaries.txt`
   (id do filme + resumo da história) e `movie.metadata.tsv` (id do filme +
   título, data, gêneros, etc.) e junta os dois pelo id do filme
   (`wiki_movie_id`), formando uma linha por filme com resumo + metadados.
3. Limpa o texto do resumo: remove tags HTML que sobraram do scraping
   original, converte pra minúsculo, normaliza caracteres Unicode esquisitos
   (ver "decisão importante" abaixo).
4. Tokeniza o texto limpo com NLTK — ou seja, quebra a frase em uma lista de
   palavras/pontuação separadas (`"o robô ama."` vira `["o", "robô", "ama",
   "."]`). Isso é feito UMA vez aqui e reaproveitado por todo o resto do
   projeto.
5. Salva tudo isso em `data/processed/movies.parquet`.
6. Faz o mesmo tipo de leitura pra `character.metadata.tsv` (um arquivo com
   um personagem/ator por linha) e salva separado em
   `data/processed/characters.parquet`.

### Por que essa abordagem

- **Por que Parquet e não CSV?** Parquet é um formato binário "colunar"
  (guarda cada coluna junto, comprimida) que preserva tipos de dado
  corretamente — inclusive listas (como a coluna `tokens`, que é uma lista de
  palavras por filme). Um CSV guardaria isso como uma string tipo
  `"['o', 'robô', 'ama']"` que cada pessoa teria que decodificar de novo do
  jeito dela. Parquet também é bem mais rápido de ler pra arquivos grandes.
- **Por que tokenizar aqui e não deixar cada método de busca tokenizar o
  próprio jeito?** Tokenizar 42 mil textos é um trabalho que custa tempo
  (ver complexidade abaixo). Se cada um dos 4 métodos de busca tokenizasse
  de novo por conta própria, esse trabalho seria repetido 4 vezes à toa — e
  pior, cada método poderia tokenizar de um jeito ligeiramente diferente,
  fazendo a comparação entre eles ficar injusta (um método "ganhando" só
  porque tokenizou melhor, não porque busca melhor). Fazer uma vez só aqui
  garante que todo mundo usa exatamente o mesmo vocabulário de base.
- **Por que `characters.parquet` é um arquivo separado?** Um filme pode ter
  dezenas de personagens. Se juntássemos tudo numa tabela só, o resumo do
  filme apareceria repetido uma vez pra cada personagem — desperdiçando
  espaço e criando duplicação de dados sem necessidade. Os dois arquivos
  compartilham a coluna `wiki_movie_id`, então dá pra juntar (merge) na hora
  que precisar, sem pagar esse custo o tempo todo.

### Complexidade

- **Download/extração:** custo fixo, não depende do tamanho do que já foi
  processado (roda uma vez só, o script pula essa etapa se os arquivos já
  existirem).
- **Parse + limpeza + tokenização:** O(n · L), onde n = número de filmes
  (~42 mil) e L = tamanho médio do resumo em caracteres/tokens.
  **O que isso significa na prática:** se o corpus dobrasse de tamanho (84
  mil filmes), o tempo de pré-processamento também dobraria — é uma relação
  direta (linear), sem surpresas. Isso é perfeitamente aceitável aqui porque
  esse processamento roda **uma vez só, offline**, não é algo que o usuário
  fica esperando toda vez que faz uma busca.

### Decisões importantes de implementação (pra saber explicar se perguntarem)

- **Bug de encoding corrigido:** o corpus tem alguns caracteres Unicode
  "quebrados" (emojis que viraram *surrogates* inválidos, tipo um emoji que
  foi cortado ao meio na hora de gerar o dataset original). Isso quebrava a
  exportação pro Parquet com um erro de encoding. A correção (`strip_surrogates`)
  remove esses caracteres inválidos sem afetar o resto do texto.
- **Idempotência:** o script verifica se o dataset já foi baixado/extraído
  antes de baixar de novo — rodar o script duas vezes não duplica trabalho.

### Limitações conhecidas

- Não faz stemming/lemmatização (não reduz "correndo" e "correu" à mesma
  raiz) — os métodos de busca leve (TF-IDF, Word2Vec) usam as palavras como
  vieram do NLTK.
- A tokenização mantém pontuação como token separado (`"."`, `","`, etc.) —
  os métodos de busca decidem por conta própria se filtram isso ou não.
- O corpus é majoritariamente em inglês; não há tratamento especial pra
  outros idiomas presentes nos resumos.

---

## Pessoa 2 — Busca leve (TF-IDF e Word2Vec)

**Código:** `src/search/cosine_baseline.py` (Método 1) e
`src/search/word2vec_search.py` (Método 2)

### O que o código faz, passo a passo

**Método 1 — TF-IDF + similaridade de cosseno:**
1. Cada sinopse vira um vetor numérico gigante, com uma posição pra cada
   palavra do vocabulário do corpus inteiro (ex: 20 mil posições). O valor em
   cada posição é o "TF-IDF" da palavra naquele documento — basicamente
   "quantas vezes essa palavra aparece aqui, mas penalizando palavras que
   aparecem em quase todo documento" (tipo "o", "de", "um" — pouco
   informativas — versus "robô", "detetive" — bem mais informativas).
2. A pergunta do usuário vira um vetor do mesmo jeito.
3. A busca compara a pergunta contra TODOS os documentos, calculando a
   "similaridade de cosseno" (basicamente: o quanto os vetores apontam na
   mesma direção) entre a pergunta e cada filme.
4. Pega os `top_k` filmes com maior similaridade.

**Método 2 — Word2Vec médio + similaridade de cosseno:**
1. Treina um modelo Word2Vec **no próprio corpus** (não baixa um pronto):
   o modelo aprende, olhando quais palavras aparecem perto de quais outras
   em milhões de frases, a representar cada palavra como um vetor denso
   pequeno (ex: 100 números), de um jeito que palavras usadas em contextos
   parecidos ficam com vetores parecidos.
2. Cada sinopse vira UM vetor: a média dos vetores de todas as suas palavras.
3. A busca funciona igual ao Método 1 (cosseno contra todos os documentos),
   só que agora os vetores são densos e bem menores.

### Por que essa abordagem

- **Por que TF-IDF como baseline?** É o método clássico e mais simples de
  Recuperação de Informação — serve de "linha de base" pra saber se os
  métodos mais sofisticados (Word2Vec, SBERT) realmente valem o custo extra
  que eles têm.
- **Por que Word2Vec é considerado "mais leve" que SBERT (Pessoa 3), mas
  ainda assim melhor que TF-IDF puro?** TF-IDF só entende **sobreposição
  exata de palavras** — se a pergunta usa "carro" e o resumo usa "automóvel",
  TF-IDF não vê relação nenhuma entre eles. Word2Vec aprende que palavras
  usadas em contextos parecidos (tipo "carro" e "automóvel" aparecendo perto
  de "dirigir", "estrada", etc.) devem ter vetores parecidos — então ele
  consegue casar sinônimos e palavras relacionadas, mesmo sem sobreposição
  exata. Isso é mais "inteligente" que TF-IDF, mas ainda mais simples e
  barato de treinar/rodar que um transformer como o SBERT da Pessoa 3 (que
  entende a frase inteira, não só palavras isoladas).
- **Por que treinar Word2Vec do zero no corpus, em vez de baixar um
  pré-treinado?** Funciona 100% offline (sem baixar nenhum modelo externo),
  o vocabulário fica exatamente adequado ao domínio (sinopses de filme, não
  um vocabulário genérico de internet), e é mais barato em memória.
- **Por que os dois métodos usam busca "força bruta" (comparar contra todo
  mundo)?** Porque ainda não têm nenhuma estrutura de indexação que evite
  isso (é exatamente esse ponto que a Pessoa 3 ataca com FAISS — embora,
  como vamos ver, o jeito que foi implementado acabou não resolvendo esse
  problema específico).

### Complexidade

**Método 1 (TF-IDF):**
- Indexação: O(n · L) pra tokenizar e contar palavras + montar a matriz
  TF-IDF esparsa.
- Busca por query: O(n · d) no pior caso, onde d = tamanho do vocabulário
  (~20 mil). Na prática, como os vetores são esparsos (a maioria das
  posições é zero), o custo real é proporcional ao número de palavras que
  realmente aparecem em cada documento, não a d inteiro.
- Seleção dos top-k: O(n) esperado (usa `np.argpartition`, que separa os k
  maiores sem ordenar todo mundo) + O(k log k) só pra ordenar os k finais.

**Método 2 (Word2Vec):**
- Indexação (treino do Word2Vec): bem mais cara que o Método 1 —
  aproximadamente O(n · L · w · d), onde w = tamanho da janela de contexto e
  d = dimensão do embedding (100). É o termo mais pesado do método.
- Busca por query: O(n · d), igual em formato ao Método 1, mas d aqui é bem
  menor (100 contra ~20 mil do TF-IDF) — então, na prática, a busca em si é
  **mais rápida** que a do Método 1, mesmo o formato assintótico sendo "o
  mesmo tipo" de custo.

**O que "O(n · d)" significa na prática, pros dois métodos:** se o número de
filmes (n) dobrar, o tempo de busca por pergunta também dobra — cresce de
forma linear e direta, sem economia de escala. Isso é a limitação que a
Pessoa 3 tenta atacar (só que, como o índice usado por ela é `IndexFlatIP` e
não HNSW, o Método 3 acaba tendo o **mesmo formato de custo na busca**,
como explicado na seção da Pessoa 3 — vale a pena ler os dois juntos).

### Decisões importantes de implementação

- **Interface comum (`BaseSemanticSearch`):** todos os métodos de busca
  seguem a mesma "receita" (`fit()` pra indexar, `search()` pra buscar,
  devolvendo sempre um `SearchResult` com `wiki_movie_id`, `title`, `score`,
  `snippet`). Isso é o que permite comparar os 4 métodos de forma uniforme
  no benchmark, sem precisar de código especial pra cada um.
- **Reaproveita os tokens da Pessoa 1:** em vez de tokenizar a sinopse de
  novo, o Word2Vec usa direto a coluna `tokens` já pronta em
  `movies.parquet` — evita repetir um trabalho O(n·L) que já foi feito uma
  vez.
- **`argpartition` em vez de ordenar tudo:** pegar os top-k sem ordenar o
  vetor de scores inteiro é uma otimização que evita um O(n log n)
  desnecessário quando só interessam os k melhores.

### Limitações conhecidas

- Os dois métodos são **força bruta**: não escalam bem pra corpora muito
  maiores que o atual (buscar sempre compara contra TODOS os documentos).
- TF-IDF não entende sinônimos/paráfrases — só sobreposição literal de
  palavras.
- Word2Vec Average perde a ordem das palavras (é uma "média", igual o
  bag-of-words, só que com vetores densos em vez de contagens esparsas) —
  não entende negação, ordem de eventos, etc.

---

## Pessoa 3 — Busca pesada (SBERT + FAISS + Reranking)

**Código:** `src/search/indexer_heavy.py`, `src/search/retriever_heavy.py`
(Método 3) e `src/search/reranker.py` (Método 4, variação com reranking)

> ⚠️ **Nota da integração final:** a análise de complexidade desta seção e
> os comentários em `indexer_heavy.py` / `retriever_heavy.py` / `reranker.py`
> foram adicionados durante a integração final do projeto (não foram escritos
> originalmente por quem implementou esta parte). Revisar e confirmar que
> faz sentido com o que você (Vinícius) implementou de fato, antes da
> apresentação — se algo aqui não bater com a sua intenção original,
> ajustar.

### O que o código faz, passo a passo

**Método 3 — SBERT + FAISS:**
1. (`indexer_heavy.py`, roda uma vez, offline) Cada sinopse passa por um
   modelo de linguagem pré-treinado (`SentenceTransformer`, modelo
   `all-MiniLM-L6-v2`) que devolve um vetor denso de 384 números
   representando o **significado da frase inteira** (não palavra por
   palavra como o Word2Vec).
2. Esses vetores são normalizados e guardados num índice FAISS
   (`IndexFlatIP`), junto com os metadados de cada filme.
3. (`retriever_heavy.py`, a cada busca) A pergunta do usuário passa pelo
   mesmo modelo, virando um vetor de 384 números.
4. O FAISS compara esse vetor contra os vetores de TODOS os filmes do
   índice (por produto interno, que equivale a similaridade de cosseno já
   que os vetores são normalizados) e devolve os `top_k` mais parecidos.

**Método 4 — Reranking com Cross-Encoder (`reranker.py`):**
1. Depois que o Método 3 devolve, digamos, os 30 candidatos mais prováveis
   (3× o `top_k` pedido), um segundo modelo (`CrossEncoder`) reordena esses
   poucos candidatos com mais cuidado: ele recebe a pergunta E a sinopse do
   candidato **juntas, numa única entrada**, e dá uma nota de relevância
   mais criteriosa pra cada par.
2. Os candidatos são reordenados por essa nova nota, e só os `top_k` finais
   (depois do reranking) são devolvidos.

### Por que essa abordagem

- **Por que SBERT em vez de Word2Vec médio?** O Word2Vec Average representa
  a frase como a MÉDIA dos vetores de cada palavra isolada — ele nunca "lê"
  a frase como um todo. O SBERT é um transformer (mesma família de
  arquitetura por trás de LLMs) treinado especificamente pra que a frase
  INTEIRA vire um vetor que captura o significado dela como unidade — ele
  entende contexto, ordem, e foi pré-treinado num volume gigantesco de texto
  da internet (transfer learning), então reconhece muito mais relações
  semânticas do que um Word2Vec treinado só nas 42 mil sinopses do projeto.
  Exemplo prático: "um robô se apaixona por um humano" e "um andróide
  desenvolve sentimentos por uma pessoa" têm quase nenhuma palavra em comum,
  mas o SBERT reconhece que são semanticamente parecidas — o Word2Vec
  Average teria muito mais dificuldade nisso.
- **Por que FAISS?** É uma biblioteca (do Facebook/Meta) especializada em
  buscar rapidamente os vizinhos mais próximos entre milhões de vetores
  densos — o "padrão da indústria" pra esse tipo de busca.
- **⚠️ Ponto importante — `IndexFlatIP` não é o que os slides originais do
  grupo descreviam:** os slides mencionavam HNSW (*Hierarchical Navigable
  Small World*), que é uma estrutura de indexação **aproximada**: ela
  organiza os vetores num grafo de "atalhos" entre vizinhos, permitindo
  pular a maior parte do corpus numa busca (parecido com "seis graus de
  separação") e achar uma resposta muito boa em tempo O(log n) esperado —
  ao custo de, ocasionalmente, não achar o vizinho mais próximo *exato*
  (recall menor que 100%).
  O código atual usa `IndexFlatIP`, que é **exato** (sempre acha os top-k
  reais de verdade, recall 100% garantido) mas **não tem nenhuma estrutura
  de atalho** — "Flat" quer dizer que o índice é só um array com todos os
  vetores guardados lado a lado, e a busca compara contra todos eles, um
  por um. **Trade-off:** exato-e-mais-lento (o que está implementado) vs.
  aproximado-e-mais-rápido (o que os slides descreviam). Pra virar a versão
  HNSW de verdade, bastaria trocar uma linha em `indexer_heavy.py`
  (`faiss.IndexFlatIP(dim)` → `faiss.IndexHNSWFlat(dim, M)`) — o resto do
  código não precisaria mudar.
- **Por que reranking com Cross-Encoder, se o SBERT já faz busca
  semântica?** O SBERT (chamado de "bi-encoder") embeda a pergunta e cada
  documento **separadamente** — isso é o que permite pré-computar os
  embeddings de todos os documentos uma vez só (na indexação) e só embedar
  a pergunta na hora da busca. O preço disso é que o modelo nunca "olha" a
  pergunta e o documento ao mesmo tempo — ele perde a chance de notar
  interações finas entre os dois textos. O Cross-Encoder resolve isso
  processando pergunta+documento JUNTOS, o que dá uma nota de relevância bem
  mais precisa — mas exatamente por isso não dá pra pré-computar nada, o
  modelo tem que rodar de novo pra cada par (pergunta, candidato), o que é
  muito mais caro por comparação. A solução (padrão bem conhecido em
  sistemas de busca/RAG) é usar os dois em sequência: o bi-encoder (barato)
  filtra rapidamente um punhado de candidatos plausíveis entre milhares, e
  só então o Cross-Encoder (caro, mas preciso) reordena esse punhado.

### Complexidade

**Indexação (`indexer_heavy.py`):**
- Encoding das sinopses pelo SentenceTransformer: uma passada do modelo por
  batch, n documentos no total — o termo mais caro da indexação, mas
  paralelizável (e usa GPU se disponível).
- `faiss.IndexFlatIP(dim)` + `index.add(embeddings)`: O(n · dim) — é
  literalmente só copiar os vetores pra um array, sem nenhum
  pré-processamento (sem clustering, sem grafo, sem quantização).

**Busca (`retriever_heavy.py`), Método 3:**
- Embedding da query: custo fixo do modelo, **não depende de n** (mesmo
  custo com 1000 ou 1 milhão de filmes no índice).
- Busca no índice Flat: compara a query contra os n vetores inteiros —
  O(n · dim), **mesmo formato de custo do Método 1 (TF-IDF)**. A diferença é
  só o "espaço" dos vetores (denso/semântico vs. esparso/lexical), não a
  ordem de crescimento da busca.

**O que "O(n · dim)" significa na prática:** dobrar o número de filmes dobra
o tempo de busca — exatamente a mesma conclusão da seção da Pessoa 2. Isso é
uma surpresa importante de destacar na apresentação: **o Método 3, do jeito
que está implementado, não é assintoticamente mais rápido que o Método 1** —
ele é semanticamente melhor (acha filmes relacionados por significado, não só
por palavra), mas não resolve o problema de escala. Só resolveria se
trocasse pra HNSW.

**Reranking (`reranker.py`), Método 4:**
- Custo por par (pergunta, candidato): uma passada completa do Cross-Encoder
  sobre os dois textos concatenados.
- Custo total: O(K), onde K = número de candidatos rerankeados (na API,
  K = 3 × top_k pedido pelo usuário) — **não é O(n)**, é O(top_k), porque o
  reranking só opera sobre o punhado de candidatos que o Método 3 já
  filtrou. **O que isso significa na prática:** dobrar o corpus inteiro
  (n) NÃO dobra o custo do reranking — só o custo da busca que roda antes
  dele. É esse "funil" (buscar barato entre milhares, refinar caro só nos
  finalistas) que torna o Cross-Encoder viável apesar de custar muito mais
  por item que o bi-encoder.
- **Limitação de implementação encontrada:** `CrossEncoder(model_name)` é
  instanciado (recarregado do zero) **a cada chamada** de
  `rerank_with_cross_encoder()`, em vez de ser carregado uma vez só e
  reaproveitado (como `HeavyRetriever` já faz com o `SentenceTransformer`).
  Isso soma um custo fixo de carregamento a cada requisição, em cima do
  custo O(K) real do reranking — no benchmark deste projeto, isso inflava a
  latência medida de ~870ms (custo real) pra quase 6 segundos (com o
  recarregamento). Vale a pena corrigir antes de uma eventual demonstração
  ao vivo com muitas buscas seguidas.

### Decisões importantes de implementação

- `IndexFlatIP` (produto interno) em vez de `IndexFlatL2` (distância
  euclidiana) — como os embeddings são normalizados (norma 1), produto
  interno e cosseno dão o mesmo resultado, e o FAISS já vem com essa
  otimização pronta.
- O reranking recebe `top_k × 3` candidatos do retriever antes de reordenar
  — dá uma margem pro Cross-Encoder "resgatar" bons candidatos que o SBERT
  sozinho não colocou exatamente no topo.

### Limitações conhecidas

- `IndexFlatIP` é exato, mas não escala sub-linearmente (mesma ordem de
  custo do Método 1) — pra ganhar velocidade de verdade em corpora maiores,
  precisaria migrar pra HNSW (ou outro índice aproximado do FAISS).
- Reload do CrossEncoder a cada chamada (ver "Complexidade" acima) —
  problema de performance, não de corretude.
- Sem testes automatizados nesta branch (diferente da Pessoa 2, que tem 18
  testes cobrindo a lógica de busca).

---

## Pessoa 4 — LLM local, API e App

**Código:** `src/llm/answer_formatter.py`, `src/api/main.py`,
`app/streamlit_app.py`

### O que o código faz, passo a passo

1. (`app/streamlit_app.py`) O usuário digita uma pergunta na interface web e
   clica em "Buscar".
2. O app faz uma requisição HTTP pra API (`GET /ask?q=...`).
3. (`src/api/main.py`) A API chama o Método 3+4 da Pessoa 3 (SBERT+FAISS,
   depois reranking) pra achar os filmes mais relevantes pra pergunta.
4. (`src/llm/answer_formatter.py`) A pergunta + os filmes encontrados são
   passados pra um modelo de linguagem local (`SmolLM2-360M-Instruct`, um
   LLM pequeno o bastante pra rodar em CPU comum, sem GPU nem chamada pra
   API paga) com uma instrução: "recomende esses filmes pro usuário, sem
   inventar filmes que não estejam na lista".
5. O LLM gera um texto de resposta em linguagem natural.
6. A API devolve a resposta + a lista dos filmes usados como fonte, em JSON.
7. O app mostra tudo isso na tela.

### Por que essa abordagem

- **Por que um LLM pequeno (360M parâmetros) e não um modelo grande tipo
  GPT?** O requisito era um LLM **local** (rodando na própria máquina, sem
  chamada de API externa/paga). Modelos grandes (bilhões de parâmetros)
  precisariam de GPU e muita memória pra rodar em tempo razoável — inviável
  num laptop de estudante. Um modelo pequeno roda em CPU em segundos, ao
  custo de gerar textos mais simples/menos sofisticados (ver limitações
  abaixo).
- **Por que essa é a etapa de "geração" do RAG?** RAG significa
  *Retrieval-Augmented Generation*: primeiro você **recupera** (retrieval)
  os documentos relevantes pra pergunta (isso é trabalho das Pessoas 2 e 3),
  depois você **gera** (generation) uma resposta em linguagem natural
  baseada NESSES documentos, em vez de deixar o LLM inventar uma resposta do
  zero usando só o que ele "decorou" no treinamento. Isso reduz alucinação
  (o LLM "viajar" e inventar informação errada) porque ele é instruído a só
  falar sobre os filmes que a busca de fato encontrou.
- **Por que decodificação gulosa (`do_sample=False`) em vez de
  amostragem/temperatura?** Testamos os dois: com amostragem
  (`do_sample=True`, temperatura 0.7), o modelo pequeno começou a alucinar
  frases sem sentido e a misturar português e inglês de forma incoerente.
  Com decodificação gulosa (sempre escolhe a palavra mais provável), o
  resultado é mais conservador — às vezes parafraseia demais a sinopse
  original em vez de "criar" uma frase nova — mas é consistentemente
  coerente. Pra um modelo desse tamanho, essa troca (menos criativo, mas
  sempre coerente) vale a pena.
- **Por que a API não fica "presa" a um método de busca específico?** A API
  foi escrita pra chamar qualquer busca que devolva filmes com
  `título`/`sinopse`/`score` — na prática ela usa o Método 3+4 (melhor
  trade-off do grupo), mas a peça de LLM (`answer_formatter.py`) não sabe
  nem se importa de onde os filmes vieram. Isso é intencional: antes das
  branches serem todas mescladas, essa API usava uma busca por palavra-chave
  provisória só pra não travar o desenvolvimento — e trocar pra busca real
  foi questão de mudar uma função, sem reescrever LLM nem app.

### Complexidade

- **Custo do LLM por resposta: O(1) em relação ao tamanho do corpus (n).**
  O modelo NUNCA vê o corpus inteiro — ele só recebe os poucos filmes (top_k,
  tipicamente 5) que a busca já filtrou. **O que isso significa na
  prática:** não importa se o banco de dados tem 42 mil filmes ou 42
  milhões — o custo de gerar a resposta final é o mesmo, porque a busca já
  fez o trabalho pesado de reduzir de "todo o corpus" pra "um punhado de
  candidatos" antes do LLM entrar em ação. O custo real depende só de
  `max_new_tokens` (quantas palavras o LLM gera) e do tamanho do próprio
  modelo — não do tamanho do banco de dados.
- **Custo da API por requisição:** soma do custo de busca (ver Pessoa 3:
  O(n·dim) pro retrieval + O(K) pro reranking) + custo O(1)-em-n do LLM. Na
  prática, o LLM e o reranking dominam o tempo total (ambos envolvem rodar
  um modelo de linguagem), não a parte de buscar no índice FAISS.

### Decisões importantes de implementação

- **Prompt com instrução anti-alucinação:** o `system prompt` instrui
  explicitamente o modelo a não inventar filmes fora da lista fornecida —
  uma técnica simples de "grounding" (ancorar a resposta nos dados reais).
- **Carregamento preguiçoso (lazy loading) do modelo:** o LLM só é carregado
  na memória na primeira vez que é usado (não no import do módulo), e depois
  fica em cache (variável global) — evita recarregar o modelo a cada
  requisição (o mesmo cuidado que faltou no reranker da Pessoa 3).
- **`MovieHit` como formato mínimo de entrada:** o formatador de resposta
  define seu próprio formato simples (`title`, `snippet`, `score`) em vez de
  depender diretamente das classes de `src/search/` — assim ele funciona
  com qualquer método de busca que sane esses três campos.

### Limitações conhecidas

- **Qualidade do texto gerado:** o modelo é pequeno, então às vezes
  parafraseia demais as sinopses originais em vez de escrever uma
  recomendação totalmente nova e natural. Funcional, mas não é uma resposta
  "polida" como a de um modelo comercial grande, que tem muito mais
  parâmetros e treino.
- **Latência do reranking herdada da Pessoa 3:** como a API usa o Método
  3+4, ela herda o problema de performance do recarregamento do
  CrossEncoder a cada chamada (~870ms de latência real, mas pode ficar mais
  lento por causa desse bug — ver seção da Pessoa 3).
- Sem testes automatizados nesta branch.

---

## Visão geral do fluxo completo

Juntando as 4 partes, o sistema funciona assim, do início ao fim:

```
1. PESSOA 1 (offline, uma vez só)
   CMU corpus (bruto) → limpeza + tokenização → movies.parquet / characters.parquet

2. PESSOA 2 e PESSOA 3 (offline, uma vez, ou quando o corpus mudar)
   movies.parquet → 4 métodos de busca "prontos pra usar":
     • Método 1: TF-IDF          (força bruta, léxico)
     • Método 2: Word2Vec médio  (força bruta, semântico simples)
     • Método 3: SBERT + FAISS   (força bruta EXATA, semântico rico)
     • Método 4: Método 3 + rerank Cross-Encoder (refina os finalistas)

3. USUÁRIO faz uma pergunta no app Streamlit (PESSOA 4)
        │
        ▼
   API FastAPI (PESSOA 4) recebe a pergunta
        │
        ├─► RETRIEVAL: chama o Método 3 (busca ampla, ~O(n·dim))
        │      depois o Método 4 (rerank só nos ~30 finalistas, O(K))
        │      → devolve os top_k filmes mais relevantes
        │
        └─► GENERATION: LLM local (PESSOA 4) recebe a pergunta + esses
               filmes e escreve uma resposta em linguagem natural,
               instruído a não inventar filmes fora da lista (custo O(1)
               em relação ao tamanho do corpus — só depende de quantos
               filmes foram retornados, não de n)
        │
        ▼
   API devolve { resposta, filmes-fonte } → app mostra na tela
```

**A ideia central de complexidade que amarra tudo:** quanto mais cedo no
pipeline, mais caro é lidar com o corpus inteiro (n); quanto mais tarde,
mais barato, porque cada etapa já filtrou o trabalho da anterior. A busca
(Métodos 1-3) é a única etapa que efetivamente olha pra todos os n filmes —
e é justamente aí que mora a maior oportunidade de otimização do projeto
(trocar `IndexFlatIP` por HNSW seria o próximo passo natural, caso o grupo
queira ir além do que já está implementado). O reranking e o LLM, por outro
lado, só lidam com um punhado de candidatos já filtrados — por isso podem
"se dar ao luxo" de usar modelos mais caros (Cross-Encoder, LLM) sem que o
sistema fique lento conforme o corpus cresce.
