"""Executa os blocos em sequência, como células de um notebook, e grava a saída de cada um."""
import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

CODIGO = Path(__file__).resolve().parent
RAIZ = CODIGO.parent
SAIDAS = RAIZ / "saidas"
SAIDAS.mkdir(exist_ok=True)
os.chdir(RAIZ)
sys.path.insert(0, str(CODIGO))

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
warnings.filterwarnings("ignore", message=".*p-value may not be accurate.*")

ambiente = {"__name__": "__main__"}
for bloco in sorted(CODIGO.glob("b[0-9][0-9]_*.py")):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(bloco.read_text(encoding="utf-8"), bloco.name, "exec"), ambiente)
    (SAIDAS / f"{bloco.stem}.txt").write_text(buffer.getvalue(), encoding="utf-8")
    print(f"=== {bloco.name} ===\n{buffer.getvalue()}")

exec(compile((CODIGO / "exportar_tabelas.py").read_text(encoding="utf-8"), "exportar_tabelas.py", "exec"), ambiente)
