"""Sobe ao Hub as PROVAS da comparacao com audio do LTX 2.5: os tres MP4 (video + audio), os tres
FLAC sem perda, a folha de contato com espectrogramas, o JSON das metricas e o README refeito.

Recusa se qualquer prova faltar: um card que descreve um video que nao esta no repo e o erro que
o dono apontou, so que na outra direcao.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot"
ROOT = Path(r"F:\COMFY_PORTABLE")
OUT = ROOT / "ComfyUI" / "output"
CARD = ROOT / "bench" / "hf" / "ltx25-22b-w4a8"

provas = {
    "README.md": CARD / "README.md",
    "LICENSE_LTX_2x_COMMUNITY.txt": CARD / "LICENSE_LTX_2x_COMMUNITY.txt",
    "av/contato_av.png": ROOT / "bench" / "ltx25" / "av" / "contato_av.png",
    "av/comparacao_av.json": ROOT / "bench" / "ltx25" / "av" / "comparacao_av.json",
    "av/bf16_original.mp4": OUT / "ltx25av_bf16_av_00001_.mp4",
    "av/int8_lightricks.mp4": OUT / "ltx25av_int8_av_00001_.mp4",
    "av/w4a8_this_file.mp4": OUT / "ltx25av_w4a8_av_00001_.mp4",
    "av/bf16_original.flac": OUT / "ltx25av_bf16_audio_00001.flac",
    "av/int8_lightricks.flac": OUT / "ltx25av_int8_audio_00001.flac",
    "av/w4a8_this_file.flac": OUT / "ltx25av_w4a8_audio_00001.flac",
}
faltam = [k for k, v in provas.items() if not v.exists()]
if faltam:
    sys.exit(f"faltam provas, nao subo nada: {faltam}")

api = HfApi()
for remoto, local in provas.items():
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remoto, repo_id=REPO, repo_type="model",
                    commit_message=f"Audio+video evidence: {remoto}")
    print(f"  subiu {remoto}  ({local.stat().st_size / 2**20:.1f} MiB)", flush=True)

no_hub = set(api.list_repo_files(REPO, repo_type="model"))
falta_no_hub = [k for k in provas if k not in no_hub]
print("LTX25_AV_UPLOAD_OK" if not falta_no_hub else f"FALTA NO HUB: {falta_no_hub}", flush=True)
