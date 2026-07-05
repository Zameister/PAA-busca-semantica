"""
Faz os testes deste diretório serem coletáveis com `pytest` rodado da raiz do
repo. Os módulos aqui (corpus_loader, cosine_baseline, ...) se importam entre
si por nome direto (ex: `from corpus_loader import ...`), então precisam de
src/search/ no sys.path — o que já acontece ao rodar cada arquivo como script,
mas não ao rodar `pytest` a partir da raiz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
