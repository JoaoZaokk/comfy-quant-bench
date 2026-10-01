"""Limpeza de 2026-09-14 ("deixa so os originais + o melhor quant de cada modelo; o que foi pro HF
ta liberado pra limpar aqui").

REGRA, escrita antes de apagar:
  apaga  = (tem sidecar .quant.json, ou seja, e conversao NOSSA)
         AND (esta no Hub: mesmo nome de arquivo E mesmo tamanho E mesmo sha256 do LFS)
         AND (nao e o build escolhido do seu modelo)
  mantem = originais e terceiros (sem sidecar: nunca tocados), tudo que NAO esta no Hub, e o build
           escolhido por modelo -- o MENOR build que o card marca como usavel (o criterio do dono:
           menor tamanho com melhor qualidade e velocidade). Copia duplicada do mesmo build em outra
           raiz conta como "nao escolhido" (fica a que os comandos documentados usam).

Cada arquivo: sha256 local calculado e comparado com o lfs.sha256 do Hub ANTES do os.remove.
Qualquer divergencia = nao apaga e diz por que. Apaga peso + .quant.json + .info do mesmo stem.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

# (caminho local, repo, caminho no Hub, motivo)
APAGAR = [
    (r"D:\ComfyUI-Models\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors",
     "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot", "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors",
     "LTX 2.5: fica o W4A8 (11,66 GiB); o int8+ConvRot empata com o int8 da Lightricks, que e original e fica"),
    (r"D:\ComfyUI-Models\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16_int8.safetensors",
     "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot", "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8.safetensors",
     "LTX 2.5: negativo medido (sem rotacao, 2x mais longe)"),
    (r"P:\ComfyBench\checkpoints\ltx-2.3-22b-distilled-1.1_w4a8.safetensors",
     "JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot", "ltx-2.3-22b-distilled-1.1_w4a8.safetensors",
     "LTX 2.3: copia duplicada; fica W:\\ltx-2.3\\ltx-2.3-22b-distilled-1.1_w4a8_W.safetensors, que os comandos usam"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models\krea2_turbo_mixed.safetensors",
     "JoaoZaokk/Krea-2-Turbo-W4A4-ConvRot", "krea2_turbo_mixed.safetensors",
     "Krea 2: fica o W4A4 (7,50 GiB, menor e mais rapido, 40 renders ok)"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\text_encoders\gemma_3_12B_it_heretic_w4a4_smooth.safetensors",
     "JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8", "w4a4/gemma_3_12B_it_heretic_w4a4_smooth.safetensors",
     "Gemma heretic: fica o W4A8; os W4A4 mudam a cena"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\text_encoders\gemma_3_12B_it_heretic_w4a4_convrot.safetensors",
     "JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8", "w4a4/gemma_3_12B_it_heretic_w4a4_convrot.safetensors",
     "Gemma heretic: fica o W4A8; os W4A4 mudam a cena"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\text_encoders\qwen3vl_4b_w4a4_convrot.safetensors",
     "JoaoZaokk/Qwen3-VL-4B-W4A8-ConvRot", "qwen3vl_4b_w4a4_convrot.safetensors",
     "Qwen3-VL 4B: fica o W4A8 (2,5x mais fiel)"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models\zimage_turbo_mixed.safetensors",
     "JoaoZaokk/Z-Image-Turbo-W4A4-ConvRot", "zimage_turbo_mixed.safetensors",
     "Z-Image Turbo: fica o W4A4 (menor, mais rapido, usavel)"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models\zimage-v2-mixed.safetensors",
     "JoaoZaokk/Beyond-Reality-Z-Image-v2-W4A4-ConvRot", "zimage-v2-mixed.safetensors",
     "Beyond-Reality Z-Image v2: fica o W4A4 (menor, mais rapido, usavel)"),
    (r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models\zimage_deturbo_mixed.safetensors",
     "JoaoZaokk/Z-Image-De-Turbo-W4A4-ConvRot", "zimage_deturbo_mixed.safetensors",
     "Z-Image De-Turbo: fica o W4A4 (menor, mais rapido, usavel)"),
]


def sha256_de(p: Path, bloco=64 * 2**20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bloco)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    api = HfApi()
    log = open(".scratch/limpeza_2026-09-14.log", "a", encoding="utf-8")
    def say(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    say(f"=== limpeza {time.strftime('%Y-%m-%d %H:%M:%S')}")
    liberado = 0
    for local, repo, hubpath, motivo in APAGAR:
        p = Path(local)
        side = p.with_suffix(".quant.json")
        info = p.with_suffix(".info")
        if not p.exists():
            say(f"AUSENTE  {p}"); continue
        if not side.exists():
            say(f"SEM SIDECAR, NAO APAGO (pode ser original): {p}"); continue
        if p.name != Path(hubpath).name:
            say(f"NOME DIFERE DO HUB, NAO APAGO: {p.name} vs {hubpath}"); continue
        infos = api.get_paths_info(repo, [hubpath], repo_type="model", expand=True)
        if not infos or getattr(infos[0], "lfs", None) is None:
            say(f"SEM LFS NO HUB, NAO APAGO: {repo}:{hubpath}"); continue
        lfs = infos[0].lfs
        tam = p.stat().st_size
        if tam != lfs.size:
            say(f"TAMANHO DIFERE, NAO APAGO: {p} local {tam} hub {lfs.size}"); continue
        t0 = time.time()
        sha = sha256_de(p)
        if sha != lfs.sha256:
            say(f"SHA DIFERE, NAO APAGO: {p} local {sha[:16]} hub {lfs.sha256[:16]}"); continue
        say(f"ok  {p.name}  {tam/2**30:.2f} GiB  sha {sha[:12]} = hub ({time.time()-t0:.0f} s)  -- {motivo}")
        for alvo in (p, side, info):
            if alvo.exists():
                os.remove(alvo)
                say(f"    apagado {alvo}")
        liberado += tam
    say(f"liberado {liberado/2**30:.2f} GiB")
    say("NAO COBERTO: so a lista acima; nada sem sidecar foi olhado; renders, embeddings, latentes e logs ficam.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
