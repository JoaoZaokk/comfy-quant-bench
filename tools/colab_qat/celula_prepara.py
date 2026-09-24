"""Celula (colab exec) para VM NOVA: confere o diffusers da VM e baixa o snapshot do klein. NÃO lança nada.

O `celula_lanca.py` baixa E lança um treino; numa VM nova para a fila, o treino vem depois, pela
`celula_lanca_fila.py`, e cada braço grava o próprio professor. Aqui só o que é comum a todos:
o pipeline do klein em /content/klein4b (repo apache-2.0, baixa sem token).
"""
from pathlib import Path

import diffusers
from huggingface_hub import snapshot_download

print(f"diffusers da VM {diffusers.__version__}, Flux2KleinPipeline "
      f"{'presente' if hasattr(diffusers, 'Flux2KleinPipeline') else 'AUSENTE'}")
Path("/content/qat").mkdir(parents=True, exist_ok=True)
raiz = snapshot_download("black-forest-labs/FLUX.2-klein-4B", token=False, local_dir="/content/klein4b",
                         allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                                         "vae/*", "transformer/*"])
prof = Path(raiz) / "transformer" / "diffusion_pytorch_model.safetensors"
print("snapshot", raiz, "professor", prof.stat().st_size)
