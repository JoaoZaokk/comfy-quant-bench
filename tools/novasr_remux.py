"""Passa o NovaSR no audio de um video pronto e remonta o mp4 com o video copiado bit a bit (so o audio muda).

Para comparar A/B o chiado do LTX sem gerar de novo: o NovaSR reamostra para 16 kHz (descarta tudo acima de 8 kHz),
converte estereo em mono e recria a banda alta ate 48 kHz. Nao escuta nem analisa o conteudo.
Uso: python -s tools/novasr_remux.py <entrada.mp4> [saida.mp4]   (padrao: <entrada>_novasr.mp4)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

RAIZ = Path(__file__).resolve().parents[1]
NODE = RAIZ / "ComfyUI" / "custom_nodes" / "ComfyUI-NovaSR"
MODELO = RAIZ / "ComfyUI" / "models" / "NovaSR" / "NovaSR.safetensors"
FFMPEG = os.environ.get("FFMPEG", r"C:\ffmpeg\bin\ffmpeg.exe")


def main() -> int:
    ent = Path(sys.argv[1])
    sai = Path(sys.argv[2]) if len(sys.argv) > 2 else ent.with_name(ent.stem + "_novasr.mp4")
    if sai.exists():
        print(f"RECUSADO: {sai} ja existe")
        return 1
    sys.path.insert(0, str(NODE))
    import novasr_node as nv
    nv.get_novasr_model_path = lambda: str(MODELO.parent)
    with tempfile.TemporaryDirectory() as td:
        wav_in, wav_out = Path(td) / "in.wav", Path(td) / "out.wav"
        subprocess.run([FFMPEG, "-v", "error", "-i", str(ent), "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_f32le",
                        str(wav_in)], check=True)
        x, sr = sf.read(wav_in, dtype="float32", always_2d=True)
        aud = {"waveform": torch.from_numpy(x.T.copy()).unsqueeze(0), "sample_rate": sr}
        out, _ = nv.NovaSRNode().upscale_audio(aud, MODELO.name, True, True, False)
        y = out["waveform"].squeeze(0).numpy().T
        pico = float(np.abs(y).max())
        if pico > 1.0:  # sem clipping no AAC
            y = y / pico
        sf.write(wav_out, y, out["sample_rate"], subtype="FLOAT")
        tmp = sai.with_suffix(".partial.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-i", str(ent), "-i", str(wav_out), "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(tmp)], check=True)
        os.replace(tmp, sai)
    print(f"ok {sai}  (entrada {sr} Hz estereo -> NovaSR 16k->48k mono duplicado; pico {pico:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
