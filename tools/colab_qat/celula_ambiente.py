"""Celula 0 (colab exec): o ambiente da VM serve para o QAT do klein? Nao instala nada, so mede.

Rodar primeiro numa sessao CPU (nao gasta GPU) e depois de novo na GPU. Responde, cada item com
o que foi executado: versoes de torch/transformers/torchao/diffusers, se `AdamW8bit` aceita
`bf16_stochastic_round`, se o Qwen3 do text encoder carrega no transformers da VM, se o repo do
klein baixa SEM token (licenca apache-2.0, nao gated -- conferir aqui, nao supor), GPU e memoria.
"""
import importlib, json, shutil, sys
r = {"python": sys.version.split()[0]}
for m in ("torch", "transformers", "torchao", "diffusers", "safetensors", "huggingface_hub", "bitsandbytes"):
    try:
        r[m] = importlib.import_module(m).__version__
    except Exception as e:  # noqa: BLE001
        r[m] = f"AUSENTE {type(e).__name__}"
try:
    import inspect, torchao.optim as o
    r["AdamW8bit_sr"] = "bf16_stochastic_round" in str(inspect.signature(o.AdamW8bit))
except Exception as e:  # noqa: BLE001
    r["AdamW8bit_sr"] = f"FALHOU {type(e).__name__}: {e}"
try:
    from transformers import Qwen3ForCausalLM  # noqa: F401
    r["qwen3"] = "ok"
except Exception as e:  # noqa: BLE001
    r["qwen3"] = f"FALHOU {type(e).__name__}"
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("black-forest-labs/FLUX.2-klein-4B", "model_index.json", token=False)
    r["hf_sem_token"] = json.load(open(p))["_class_name"]
except Exception as e:  # noqa: BLE001
    r["hf_sem_token"] = f"FALHOU {type(e).__name__}: {str(e)[:120]}"
try:
    import torch
    r["cuda"] = torch.cuda.is_available()
    if r["cuda"]:
        r["gpu"] = torch.cuda.get_device_name(0)
        r["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        r["cc"] = torch.cuda.get_device_capability(0)
except Exception as e:  # noqa: BLE001
    r["cuda"] = f"FALHOU {e}"
r["disco_content_gib"] = round(shutil.disk_usage("/content").free / 2**30, 1)
r["ram_gib"] = round(int(open("/proc/meminfo").read().split()[1]) / 2**20, 1)
r["drive_montado"] = shutil.os.path.isdir("/content/drive/MyDrive")
print("AMBIENTE=" + json.dumps(r))
