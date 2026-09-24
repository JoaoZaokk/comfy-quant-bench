"""Encode REAL do text encoder LTX com os residuos quantizados (embedding do Gemma + projecao) contra
o par de producao (Gemma W4A8 + projecao BF16), pelo mesmo caminho do no `LTXV2AVTextEncoderLoader`:
`comfy.sd.load_clip([gemma, projecao], clip_type=LTXV)`, travas do TE mantidas como em producao.

    python_embeded\\python.exe -s tools/probe_te_residuos_encode.py --device 1

Cada braco roda num processo proprio (RAM e VRAM limpas entre eles), na placa pedida via
CUDA_VISIBLE_DEVICES. O braco `base` roda duas vezes: a diferenca base x base2 e o piso de ruido
da propria GPU, e nenhum erro abaixo dele significa nada. Toma o lock GPU (BenchGuard).

Mede, por braco: tamanho do modelo carregado, pico de VRAM do encode, tempo do encode (mediana
de 3, depois de aquecer) e quantas camadas de embedding/projecao subiram como QuantizedTensor.
Compara o condicionamento (parte de video e parte de audio separadas) contra `base`.

NAO COBERTO: fidelidade do condicionamento nao e qualidade de render; prompts neutros (SFW), sem
os prompts reais de uso; uma placa.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TE = ROOT / "ComfyUI" / "models" / "text_encoders"
CK = Path("P:/ComfyBench/checkpoints")
BRACOS = {
    "base": ("gemma_3_12B_it_heretic_w4a8.safetensors", "ltx-2.3_text_projection_bf16.safetensors"),
    "base2": ("gemma_3_12B_it_heretic_w4a8.safetensors", "ltx-2.3_text_projection_bf16.safetensors"),
    "int8": ("gemma_3_12B_it_heretic_w4a8_embint8.safetensors", "ltx-2.3_text_projection_int8.safetensors"),
    "fp8": ("gemma_3_12B_it_heretic_w4a8_embfp8.safetensors", "ltx-2.3_text_projection_fp8.safetensors"),
    # decomposicao: uma peca quantizada de cada vez
    "emb_int8": ("gemma_3_12B_it_heretic_w4a8_embint8.safetensors", "ltx-2.3_text_projection_bf16.safetensors"),
    "proj_int8": ("gemma_3_12B_it_heretic_w4a8.safetensors", "ltx-2.3_text_projection_int8.safetensors"),
    "emb_fp8": ("gemma_3_12B_it_heretic_w4a8_embfp8.safetensors", "ltx-2.3_text_projection_bf16.safetensors"),
    "proj_fp8": ("gemma_3_12B_it_heretic_w4a8.safetensors", "ltx-2.3_text_projection_fp8.safetensors"),
}
PROMPTS = [
    "A red apple on a weathered wooden table, soft window light.",
    "Um homem idoso conserta um relógio antigo na bancada, a câmera se aproxima devagar, "
    "ele diz em voz baixa: \"quase pronto\". Som ambiente de oficina, tique-taque ao fundo.",
    "Wide shot of a rain-soaked neon alley at night. A woman in a yellow raincoat walks toward the "
    "camera, stops, looks up and says clearly: \"We're late, come on.\" Rain patters on metal roofs, "
    "distant traffic hum, a dog barks twice. The camera slowly dollies back as she keeps walking, "
    "neon signs flicker in pink and blue, reflections ripple across the wet asphalt.",
]

FILHO = r'''
import json, statistics, sys, time
sys.path.insert(0, "ComfyUI"); sys.argv = ["main.py"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, comfy.sd
from comfy_kitchen.tensor.base import QuantizedTensor
G, P, PROMPTS, OUT = %(G)r, %(P)r, json.loads(%(PROMPTS)r), %(OUT)r
clip = comfy.sd.load_clip(ckpt_paths=[G, P], clip_type=comfy.sd.CLIPType.LTXV)
rep = {"model_size_gib": clip.patcher.model_size() / 2**30}
alvo = {n: type(m.weight).__name__ for n, m in clip.cond_stage_model.named_modules()
        if n.endswith("embed_tokens") or n.endswith("aggregate_embed")}
rep["camadas"] = alvo
def encode(p):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = clip.encode_from_tokens_scheduled(clip.tokenize(p))
    torch.cuda.synchronize()
    return out[0][0].float().cpu(), (time.perf_counter() - t0) * 1e3
encode(PROMPTS[0])
torch.cuda.reset_peak_memory_stats()
conds, tempos = [], []
for p in PROMPTS:
    ts = []
    for _ in range(3):
        c, t = encode(p); ts.append(t)
    conds.append(c); tempos.append(statistics.median(ts))
rep["tempo_ms"] = tempos
rep["pico_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
rep["carregado_gib"] = clip.patcher.loaded_size() / 2**30
torch.save(conds, OUT + ".pt")
open(OUT + ".json", "w").write(json.dumps(rep))
print(json.dumps(rep))
'''


def compara(a, b) -> dict:
    import torch
    out = {}
    for nome, fatia in (("video", slice(0, 4096)), ("audio", slice(4096, None))):
        num = den = 0.0
        cos = []
        for x, y in zip(a, b):
            x, y = x[..., fatia].flatten(0, -2), y[..., fatia].flatten(0, -2)
            num += (x - y).pow(2).sum().item()
            den += y.pow(2).sum().item()
            cos.append(torch.nn.functional.cosine_similarity(x, y, dim=-1).min().item())
        out[nome] = {"rel": (num / den) ** 0.5, "cos_min_token": min(cos)}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="1", help="indice fisico da placa (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--bracos", nargs="+", default=None, help="subconjunto (base e sempre incluido)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "bench" / "te_residuos")
    p.add_argument("--teto-gib", type=float, default=None,
                   help="teto de ocupacao do BenchGuard (padrao dele: 2 GiB por placa). Subir so' quando a "
                        "ocupacao for de programas do desktop na OUTRA placa; fica registrado no resumo")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "tools"))
    from _bench_guard import BenchGuard

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device)
    extra = {} if args.teto_gib is None else {"ceiling_gib": args.teto_gib}
    rel = {"_teto_gib": args.teto_gib, "_placa": args.device}
    with BenchGuard("probe_te_residuos_encode", **extra) as guarda:
        if guarda.refused:
            print("recusado:", guarda.refused)
            return 1
        escolhidos = ["base"] + [b for b in (args.bracos or BRACOS) if b != "base"]
        for braco in escolhidos:
            g, pj = BRACOS[braco]
            out = str(args.out_dir / braco)
            src = FILHO % {"G": str(TE / g), "P": str(CK / pj), "PROMPTS": json.dumps(PROMPTS), "OUT": out}
            r = subprocess.run([sys.executable, "-s", "-c", src], cwd=ROOT, env=env,
                               capture_output=True, text=True)
            linha = [x for x in r.stdout.splitlines() if x.startswith("{")]
            print(braco, "rc", r.returncode, linha[-1] if linha else r.stderr[-2000:], flush=True)
            if r.returncode:
                return 1
            rel[braco] = json.loads(linha[-1])

    import torch
    base = torch.load(args.out_dir / "base.pt")
    for braco in BRACOS:
        if braco == "base" or braco not in rel:
            continue
        rel[braco]["vs_base"] = compara(torch.load(args.out_dir / f"{braco}.pt"), base)
        print(braco, "vs base", rel[braco]["vs_base"])
    (args.out_dir / ("resumo.json" if args.bracos is None else f"resumo_{'_'.join(args.bracos)}.json")).write_text(json.dumps(rel, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
