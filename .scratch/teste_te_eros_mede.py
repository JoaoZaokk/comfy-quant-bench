"""Compara numericamente os videos do teste de text encoder (teste_te_eros.py) sem exibir nada:
PSNR por quadro (reduzido a 256 px de largura) e, no audio, distancia de espectro log-mel e
correlacao da forma de onda. O piso de ruido e base x o render anterior do mesmo grafo
(I2V_DMD_tiled_00001), feito em outra execucao do servidor.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path("ComfyUI/output/Eros")
FF = "ffmpeg"


def quadros(p: Path, w=256):
    info = json.loads(subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(p)],
                                     capture_output=True, text=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    h = int(round(int(v["height"]) * w / int(v["width"]) / 2) * 2)
    raw = subprocess.run([FF, "-v", "quiet", "-i", str(p), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def audio(p: Path, sr=16000):
    raw = subprocess.run([FF, "-v", "quiet", "-i", str(p), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def logmel(x, sr=16000):
    x = torch.from_numpy(x.copy())
    S = torch.stft(x, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs() ** 2
    f = torch.linspace(0, sr / 2, S.shape[0])
    mel = 2595 * torch.log10(1 + f / 700)
    edges = torch.linspace(mel[0], mel[-1], 66)
    fb = torch.stack([((mel >= edges[i]) & (mel < edges[i + 2])).float() for i in range(64)])
    return torch.log10(fb @ S + 1e-8)


def compara(a: Path, b: Path) -> dict:
    fa, fb = quadros(a), quadros(b)
    n = min(len(fa), len(fb))
    mse = ((fa[:n] - fb[:n]) ** 2).mean(axis=(1, 2, 3))
    psnr = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9))
    xa, xb = audio(a), audio(b)
    m = min(len(xa), len(xb))
    out = {"quadros": [len(fa), len(fb)], "psnr_medio": float(psnr.mean()), "psnr_min": float(psnr.min()),
           "psnr_q1": float(psnr[: n // 4].mean()), "psnr_q4": float(psnr[-n // 4:].mean())}
    if m:
        la, lb = logmel(xa[:m]), logmel(xb[:m])
        out["audio_logmel_l1"] = float((la - lb).abs().mean())
        out["audio_corr_onda"] = float(np.corrcoef(xa[:m], xb[:m])[0, 1])
        out["audio_rms"] = [float(np.sqrt((xa ** 2).mean())), float(np.sqrt((xb ** 2).mean()))]
    return out


def main():
    base = OUT / "teste_te_base_00001-audio.mp4"
    pares = {"base_vs_anterior(piso)": (base, OUT / "I2V_DMD_tiled_00001-audio.mp4"),
             "fp8_vs_base": (OUT / "teste_te_fp8_00001-audio.mp4", base),
             "int8_vs_base": (OUT / "teste_te_int8_00001-audio.mp4", base),
             "int8_vs_fp8": (OUT / "teste_te_int8_00001-audio.mp4", OUT / "teste_te_fp8_00001-audio.mp4")}
    res = {}
    for nome, (a, b) in pares.items():
        if a.exists() and b.exists():
            res[nome] = compara(a, b)
            print(nome, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[nome].items()}), flush=True)
        else:
            print(nome, "faltando", a.exists(), b.exists())
    Path("bench/te_residuos/render_eros.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
