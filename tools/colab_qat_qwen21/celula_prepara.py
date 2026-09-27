"""Celula (colab exec): prepara a VM para o QAT Qwen 2.1 -- descompacta o pacote `comfy` (v0.37.4), confere os imports
que o modelo precisa e, com PREPARA_BAIXA=1 no config, baixa o BF16 da Comfy-Org. Nao instala nada pesado."""
import importlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
Q = Path("/content/qatq")
cfg = json.loads((Q / "config.json").read_text())
C = Path("/content/ComfyUI")
if not (C / "comfy").is_dir():
    C.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(Q / "comfy_v0374.zip").extractall(C)
r = {"python": sys.version.split()[0]}
# o comfy importa comfy_aimdo sem fallback; mesmas versoes da bancada, so os dois wheels, sem dependencias
for pkg, mod in (("comfy-aimdo==0.5.5", "comfy_aimdo"), ("comfy-kitchen==0.2.35", "comfy_kitchen")):
    try:
        importlib.import_module(mod)
    except Exception:  # noqa: BLE001
        rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", pkg]).returncode
        r["pip_" + mod] = rc
for m in ("torch", "numpy", "safetensors", "einops", "psutil", "scipy", "transformers", "huggingface_hub",
          "comfy_kitchen", "comfy_aimdo"):
    try:
        r[m] = getattr(importlib.import_module(m), "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        r[m] = f"AUSENTE {type(e).__name__}"
sys.path.insert(0, str(C))
import torch
import comfy.cli_args
if not torch.cuda.is_available():
    comfy.cli_args.args.cpu = True
try:
    import comfy.model_management, comfy.utils, comfy.model_detection, comfy.ops  # noqa: F401
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel  # noqa: F401
    r["comfy_import"] = "ok"
except Exception as e:  # noqa: BLE001
    import traceback
    r["comfy_import"] = f"FALHOU {type(e).__name__}: {e}"
    print(traceback.format_exc()[-2500:])
import torch
r["cuda"] = torch.cuda.is_available()
if r["cuda"]:
    r["gpu"] = torch.cuda.get_device_name(0)
    r["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
r["disco_gib"] = round(shutil.disk_usage("/content").free / 2**30, 1)
r["ram_gib"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
if cfg.get("baixa"):
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    p = hf_hub_download("Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_bf16.safetensors",
                        local_dir="/content/qwen21", token=False)
    r["dit"] = p
    r["dit_bytes"] = Path(p).stat().st_size
    r["dit_s"] = round(time.time() - t0)
print(json.dumps(r, indent=1))
