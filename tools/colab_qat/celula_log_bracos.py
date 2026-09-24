"""Celula (colab exec): cauda do qat.log de cada braco da fila (o log da fila higieniza as palavras de falha)."""
import json
from pathlib import Path
for b in json.loads(Path("/content/qat/fila.json").read_text())["bracos"]:
    lg = Path(b["dir"]) / "qat.log"
    print(f"===== {b['nome']} {lg} =====")
    print("\n".join(lg.read_text(errors="replace").splitlines()[-25:]) if lg.is_file() else "(sem log)")
