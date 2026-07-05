"""
Queries de exemplo usadas por cosine_baseline.py, word2vec_search.py e benchmark.py.

Centralizadas aqui de propósito: se cada script tivesse sua própria lista, elas
poderiam divergir com o tempo e a comparação entre métodos deixaria de ser justa
(cada um sendo testado com queries diferentes). Editem esta lista à vontade —
ela é consumida por todo o resto automaticamente.
"""

EXAMPLE_QUERIES: list[str] = [
    "a detective investigates a murder in the city",
    "a robot falls in love with a human",
    "soldiers survive a brutal war",
    "a group plans a heist to steal something valuable",
    "a monster hunts people in the woods at night",
]
