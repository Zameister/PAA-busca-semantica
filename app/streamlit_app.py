"""
streamlit_app.py — Pessoa 4: interface web mínima. Só chama a API
(src/api/main.py) e mostra a resposta do LLM + os filmes usados como fonte.
Rodar a API antes: uvicorn src.api.main:app
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.title("Busca semântica de filmes (PAA)")

query = st.text_input("O que você procura?", placeholder="ex: um robô se apaixona por um humano")
top_k = st.slider("Quantos filmes considerar", 1, 10, 5)

if st.button("Buscar") and query:
    with st.spinner("Buscando e gerando resposta..."):
        try:
            resp = requests.get(f"{API_URL}/ask", params={"q": query, "top_k": top_k}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Não consegui falar com a API em {API_URL} ({e}). Ela está rodando?")
        else:
            st.subheader("Resposta")
            st.write(data["answer"])

            st.subheader("Filmes usados como fonte")
            for movie in data["movies"]:
                st.markdown(f"**{movie['title']}** (score: {movie['score']:.2f})")
                st.caption(movie["snippet"])
