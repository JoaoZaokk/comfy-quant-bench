"""Celula (colab exec): lanca `fila_qat_ternario_klein.py` em BACKGROUND e aponta o probe para a fila.

Pre-requisitos ja' na VM (subidos com `colab upload` antes): /content/qat/fila.json,
/content/qat/fila_qat_ternario_klein.py e o `qat_ternario_klein.py` NOVO (sobrescrever o arquivo com
o replay rodando e' seguro: o Python ja' o carregou). O snapshot do klein ja' esta' em /content/klein4b
(baixado pelo `celula_lanca.py` do replay) e os shards do professor em /content/qat_run.

Troca /content/qat/config.json para {"dir": "/content/qat_fila"} -- e' esse arquivo que o
`probe_qat.py` le a cada 4 min. A partir daqui o supervisor observa a FILA, nao o replay.
"""
import json
import subprocess
import sys
from pathlib import Path

Q = Path("/content/qat")
cfg = json.loads((Q / "fila.json").read_text())
for p in (cfg["raiz"], cfg["professor"], cfg["professor_de"] + "/professor",
          cfg["professor_de"] + "/professor_holdout", str(Q / "qat_ternario_klein.py")):
    if not Path(p).exists():
        print(f"FALTA {p}")
        raise SystemExit(1)
n = len(list(Path(cfg["professor_de"], "professor").glob("p*_s1.pt")))
print(f"shards do professor (semente 1): {n}")
src = (Q / "qat_ternario_klein.py").read_text()
print("tool novo:", "prox_l1_relativo" in src and "mede_d" in src)
Path("/content/qat_fila").mkdir(parents=True, exist_ok=True)
(Q / "config.json").write_text(json.dumps({"dir": "/content/qat_fila"}))
log = open("/content/qat_fila/lancamento.log", "ab")
p = subprocess.Popen(["setsid", "nohup", sys.executable, "-u", str(Q / "fila_qat_ternario_klein.py")],
                     stdout=log, stderr=subprocess.STDOUT, cwd=str(Q), start_new_session=True)
print(f"FILA LANCADA pid={p.pid}")
