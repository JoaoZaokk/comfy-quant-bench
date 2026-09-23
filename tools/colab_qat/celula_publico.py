from huggingface_hub import HfApi
a = HfApi(); r = "JoaoZaokk/klein4b-qat-ckpt"
card = """---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
tags: [quantization, ternary, qat, work-in-progress]
---
# klein4b-qat-ckpt — training checkpoint, not a usable model

Rolling checkpoint of an ongoing quantization-aware-training (QAT) experiment: the transformer of
[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) with its 100 block Linear
layers trained as ternary (1.58-bit, absmean, group 128 on K) by distillation against the BF16 original.

`ckpt/ultimo.pt` is a raw PyTorch pickle (bf16 master weights in diffusers naming + torchao AdamW8bit
state). It is overwritten every ~25 minutes, is **not** ternary-packed, and is **not** meant to be
loaded for inference. Derived from FLUX.2-klein-4B, licensed Apache-2.0.
"""
a.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=r)
a.update_repo_settings(r, private=False)
print("publico:", not a.model_info(r).private)
