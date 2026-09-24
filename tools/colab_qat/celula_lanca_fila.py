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
import shutil
import time
for b in cfg["bracos"]:
    d = Path(b["dir"])
    # braco que morreu antes do primeiro checkpoint: recomecar do zero (senao o journal e o melhor.json
    # velhos se misturam com a corrida nova). Com checkpoint, a corrida RETOMA -- nao se apaga.
    if d.is_dir() and not (d / "ckpt" / "ultimo.pt").is_file():
        shutil.rmtree(d)
        print(f"limpo {d} (sem checkpoint)")
Path("/content/qat_fila").mkdir(parents=True, exist_ok=True)
velho = Path("/content/qat_fila/qat.log")
if velho.is_file():
    velho.rename(velho.with_name(f"qat.{time.strftime('%H%M%S')}.log"))
(Q / "config.json").write_text(json.dumps({"dir": "/content/qat_fila"}))
log = open("/content/qat_fila/lancamento.log", "ab")
p = subprocess.Popen(["setsid", "nohup", sys.executable, "-u", str(Q / "fila_qat_ternario_klein.py")],
                     stdout=log, stderr=subprocess.STDOUT, cwd=str(Q), start_new_session=True)
print(f"FILA LANCADA pid={p.pid}")
