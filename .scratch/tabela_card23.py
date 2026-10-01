"""Gera as tabelas do card do LTX 2.3 a partir dos artefatos, sem transcricao a mao:
- bench/ltx23/av/comparacao_av.json (referencia BF16 por GGUF; bracos Q6_K, W4A8, W4A4)
- .scratch/<corrida>.json (parede da corrida inteira, `segundos`)
- barra do tqdm no .scratch/comfy_8190_j.err (tempo do SAMPLER, via tools/sampler_tempo_do_log.parse),
  mapeada pela ORDEM da fila j: 1 identidade, 2-5 LoRA farol, 6-9 LoRA gatilho, 10 Q6_K, 11 W4A4, 12 BF16
- tamanho real dos arquivos carregados (stat).
Imprime markdown. Nao grava nada. No fim diz o que NAO cobre."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "tools")
from sampler_tempo_do_log import parse  # noqa: E402

ROOT = Path(".")
AV = json.loads((ROOT / "bench/ltx23/av/comparacao_av.json").read_text(encoding="utf-8"))
LOG = (ROOT / ".scratch/comfy_8190_j.err").read_text(encoding="utf-8", errors="replace")
prompts = [p for p in parse(LOG) if p["barras"]]
ORDEM = {1: "bf16"} if False else {}
# ordem da fila j (so prompts com sampler): 1 identidade w4a8, 2-5 lora farol, 6-9 lora gatilho, 10 q6k, 11 w4a4, 12 bf16
IDX = {"w4a8": 1, "q6k": 10, "w4a4": 11, "bf16": 12}
ARQ = {
    "bf16": ("BF16 original (lossless GGUF container, transformer only)", Path("C:/ComfyBench/ltx-2.3/ltx-2.3-22b-distilled-1.1-BF16.gguf")),
    "q6k": ("GGUF Q6_K, third party (transformer only)", Path("ComfyUI/models/unet/ltx-2.3-22b-distilled-1.1-Q6_K.gguf")),
    "w4a8": ("**this file, W4A8** (single-file checkpoint)", Path("P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a8.safetensors")),
    "w4a4": ("W4A4, same source (control; not published)", Path("P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a4.safetensors")),
}
RUN = {"bf16": "ltx23av_bf16", "q6k": "ltx23av_q6k_condf", "w4a8": "ltx23av_w4a8_condf", "w4a4": "ltx23av_w4a4_condf"}
BRACO = {"q6k": "GGUF Q6_K (third party)", "w4a8": "W4A8 ours", "w4a4": "W4A4 ours (control)"}


def gib(p: Path) -> str:
    return f"{p.stat().st_size / 2**30:.2f}" if p.exists() else "?"


def samp(k: str) -> tuple[str, str]:
    i = IDX[k]
    if i > len(prompts):
        return "?", "?"
    b = prompts[i - 1]["barras"]
    if not b:
        return "-", "-"
    return f"{b[0]['s_por_it']:.2f}", f"{b[0]['segundos']}"


def parede(k: str) -> str:
    j = ROOT / f".scratch/{RUN[k]}.json"
    if not j.exists():
        return "?"
    d = json.loads(j.read_text(encoding="utf-8"))
    return f"{d['segundos']:.0f}"


print("### Picture\n")
print("| arm | GiB on disk | sampler, 8 steps | whole run | MAE vs BF16 | PSNR | SSIM | motion |")
print("|---|---|---|---|---|---|---|---|")
ref = AV["referencia"]
sit, ss = samp("bf16")
print(f"| {ARQ['bf16'][0]} | {gib(ARQ['bf16'][1])} | {sit} s/it ({ss} s) | {parede('bf16')} s | — | — | — | {ref['movimento']:.2f} |")
for k in ("q6k", "w4a8", "w4a4"):
    b = AV["bracos"][BRACO[k]]
    v = b["video"]
    sit, ss = samp(k)
    print(f"| {ARQ[k][0]} | {gib(ARQ[k][1])} | {sit} s/it ({ss} s) | {parede(k)} s | {v['mae']:.2f} [{v['mae_min']:.2f}–{v['mae_max']:.2f}] | {v['psnr']:.2f} dB | {v['ssim']:.3f} | {b['movimento']:.2f} |")

print("\n### Sound\n")
print("| arm | log-mel L1 vs BF16 | SNR | spectral convergence | lag | level |")
print("|---|---|---|---|---|---|")
for k in ("q6k", "w4a8", "w4a4"):
    a = AV["bracos"][BRACO[k]]["audio"]
    print(f"| {ARQ[k][0].split(' (')[0]} | {a['logmel_l1']:.3f} | {a['snr_db']:.1f} dB | {a['conv_espectral']:.3f} | {a['lag_ms']:+.1f} ms | {a['rms_dbfs']:.1f} dBFS |")
c = AV["controles"]
print(f"| *control: silence* | {c['silencio']['logmel_l1']:.3f} | {c['silencio']['snr_db']:.1f} dB | — | — | — |")
r = c["ruido_branco_mesmo_rms"]
print(f"| *control: white noise at the reference's RMS* | {r['logmel_l1']:.3f} | {r['snr_db']:.1f} dB | {r['conv_espectral']:.3f} | — | — |")
print(f"\nreference RMS {ref['rms_dbfs']:.1f} dBFS, silence {ref['silencio_frac']*100:.0f} %, audio {ref['audio_s']} s at {ref['sr']} Hz")
print("\nNAO COBERTO: o mapeamento prompt->braco e pela ORDEM da fila j (confira o numero de prompts com barra: "
      f"{len(prompts)}); tamanho em disco e o arquivo carregado, nao o transformer isolado nos single-file; "
      "MAE/PSNR/SSIM sao distancias, nao qualidade.")
