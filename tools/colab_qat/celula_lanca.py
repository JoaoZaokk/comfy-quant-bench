"""Celula de lancamento (colab exec): instala o diffusers offline e sobe o QAT em BACKGROUND.

`colab exec` NAO passa argumento: a configuracao vem de /content/qat/config.json (subido com
`colab upload`), junto com os arquivos de `colab_ops.ARQUIVOS_QAT`. Nunca montar ambiente pesado aqui --
so o wheel puro-python do diffusers 0.38.0, sha256 conferido antes de instalar.

O processo sobe com `setsid nohup` e a celula RETORNA. Isso sozinho mata a VM em 22-26 min
(kernel ocioso): o supervisor com `probe_qat.py` a cada 4 min e OBRIGATORIO depois disto.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

Q = Path("/content/qat")
sys.path.insert(0, str(Q))
import colab_ops

colab_ops.exige([Q / f for f in colab_ops.ARQUIVOS_QAT] + [Q / "config.json"])
cfg = json.loads((Q / "config.json").read_text())
whl = Q / "diffusers-0.38.0-py3-none-any.whl"
SHA = "18e53f9e539096320470f62c6360a6fd5727ff28cffda566265316e13fcdb612"
# A VM JA TRAZ diffusers 0.40.0 com transformers 5.16.1 [MEDIDO 2026-09-22, sessao CPU]. Forcar o
# 0.38.0 daqui seria um downgrade contra um transformers com que ele nunca foi testado; entao so se
# instala o wheel se o pipeline do klein FALTAR. Consequencia registrada: o ajuste local rodou no
# 0.38.0/4.57.6 e o Colab roda 0.40.0/5.16.1 -- a avaliacao final e no ComfyUI local, igual para todos.
try:
    import diffusers
    ok = hasattr(diffusers, "Flux2KleinPipeline")
    print(f"diffusers da VM {diffusers.__version__}, Flux2KleinPipeline {'presente' if ok else 'AUSENTE'}")
except Exception:  # noqa: BLE001
    ok = False
if not ok:
    h = hashlib.sha256(whl.read_bytes()).hexdigest()
    if h != SHA:
        print(f"FALHOU: sha256 do wheel {h} != {SHA}")
        raise SystemExit(1)
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(whl)], check=False).returncode
    print(f"pip rc={rc}")
    if rc:
        raise SystemExit(rc)
d = Path(cfg["dir"])
d.mkdir(parents=True, exist_ok=True)
raiz, prof = colab_ops.snapshot_klein()
extra = []
if cfg.get("inicio_arquivo_local") and Path(cfg["inicio_arquivo_local"]).is_file():
    # Relancamento: o checkpoint de PARTIDA ja esta no disco. Baixar de novo do HF traria o que o
    # repo tiver AGORA (outra corrida sobrescreve o mesmo caminho) e trocaria a origem calado.
    extra = ["--inicia-de", cfg["inicio_arquivo_local"]]
    print("inicio (local, sem baixar)", cfg["inicio_arquivo_local"])
elif cfg.get("inicio_hf"):
    # checkpoint de partida de um repo PUBLICO: nao precisa de token nesta VM
    from huggingface_hub import hf_hub_download
    ini = hf_hub_download(cfg["inicio_hf"], "ckpt/ultimo.pt", local_dir="/content/inicio", token=False)
    print("inicio", ini, Path(ini).stat().st_size)
    extra = ["--inicia-de", ini]
cmd = [sys.executable, "-u", Q / "qat_ternario_klein.py", "--raiz", raiz, "--professor", prof,
       "--prompts", Q / "prompts_treino.txt", "--prompts-holdout", Q / "prompts_holdout.txt",
       "--dir", d, *cfg["args"], *extra]
colab_ops.lanca(cmd, d / "qat.log")
