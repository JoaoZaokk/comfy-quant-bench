"""Celula (colab exec): lanca `fila_qat_ternario_klein.py` em BACKGROUND e aponta o probe para a fila.

Pre-requisitos ja' na VM (subidos com `colab upload` antes): /content/qat/fila.json,
/content/qat/fila_qat_ternario_klein.py e os arquivos de `colab_ops.ARQUIVOS_QAT` (sobrescrever o QAT com
outra corrida rodando e' seguro: o Python ja' o carregou). O snapshot do klein ja' esta' em /content/klein4b.

Troca /content/qat/config.json para {"dir": "/content/qat_fila"} -- e' esse arquivo que o
`probe_qat.py` le a cada 4 min. A partir daqui o supervisor observa a FILA.
"""
import json
import sys
import time
from pathlib import Path

Q = Path("/content/qat")
sys.path.insert(0, str(Q))
import colab_ops

cfg = json.loads((Q / "fila.json").read_text())
# VM nova (professor_de null): cada braco grava o proprio professor; so' o pipeline e o tool precisam existir
exig = [cfg["raiz"], cfg["professor"], Q / "fila_qat_ternario_klein.py", Q / "prompts_holdout.txt",
        *[Q / f for f in colab_ops.ARQUIVOS_QAT],
        *[Q / b.get("prompts", "prompts_treino.txt") for b in cfg["bracos"]]]
if cfg.get("professor_de"):
    exig += [cfg["professor_de"] + "/professor", cfg["professor_de"] + "/professor_holdout"]
colab_ops.exige(exig)
if cfg.get("professor_de"):
    n = len([a for a in Path(cfg["professor_de"], "professor").glob("*.pt") if "_s1" in a.stem])
    print(f"shards do professor (semente 1): {n}")
for b in cfg["bracos"]:
    # braco que morreu antes do primeiro checkpoint: o journal e o melhor.json velhos nao podem se misturar
    # com a corrida nova. So' esses saem -- os shards do professor da pasta ficam. Com checkpoint, retoma.
    apagados = colab_ops.limpa_braco_sem_ckpt(Path(b["dir"]))
    if apagados:
        print(f"limpo {b['dir']} (sem checkpoint): {apagados}")
Path("/content/qat_fila").mkdir(parents=True, exist_ok=True)
for velho in (Path("/content/qat_fila/qat.log"), Path("/content/qat_fila/status.json")):
    if velho.is_file():
        velho.rename(velho.with_name(f"{velho.stem}.{time.strftime('%H%M%S')}{velho.suffix}"))
(Q / "config.json").write_text(json.dumps({"dir": "/content/qat_fila"}))
colab_ops.lanca([sys.executable, "-u", Q / "fila_qat_ternario_klein.py"], Path("/content/qat_fila/lancamento.log"))
