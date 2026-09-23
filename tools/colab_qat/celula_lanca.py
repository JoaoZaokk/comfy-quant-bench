"""Celula de lancamento (colab exec): instala o diffusers offline e sobe o QAT em BACKGROUND.

`colab exec` NAO passa argumento: a configuracao vem de /content/qat/config.json (subido com
`colab upload`). Nunca montar ambiente pesado aqui -- so o wheel puro-python do diffusers 0.38.0,
sha256 conferido antes de instalar.

O processo sobe com `setsid nohup` e a celula RETORNA. Isso sozinho mata a VM em 22-26 min
(kernel ocioso): o supervisor com `probe_qat.py` a cada 4 min e OBRIGATORIO depois disto.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
Q = Path("/content/qat")
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
        print(f"FALHOU: sha256 do wheel {h} != {SHA}"); raise SystemExit(1)
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(whl)]).returncode
    print(f"pip rc={rc}")
    if rc:
        raise SystemExit(rc)
d = Path(cfg["dir"]); d.mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
raiz = snapshot_download("black-forest-labs/FLUX.2-klein-4B", token=False,
                         local_dir="/content/klein4b", allow_patterns=[
                             "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                             "vae/*", "transformer/*"])
prof = Path(raiz) / "transformer" / "diffusion_pytorch_model.safetensors"
print("snapshot", raiz, "professor", prof.stat().st_size)
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
cmd = [sys.executable, "-u", str(Q / "qat_ternario_klein.py"), "--raiz", raiz, "--professor", str(prof),
       "--prompts", str(Q / "prompts_treino.txt"), "--prompts-holdout", str(Q / "prompts_holdout.txt"),
       "--dir", str(d), *cfg["args"], *extra]
print("CMD", " ".join(cmd))
log = open(d / "qat.log", "ab")
p = subprocess.Popen(["setsid", "nohup", *cmd], stdout=log, stderr=subprocess.STDOUT, cwd=str(Q),
                     start_new_session=True)
print(f"LANCADO pid={p.pid} log={d / 'qat.log'}")
