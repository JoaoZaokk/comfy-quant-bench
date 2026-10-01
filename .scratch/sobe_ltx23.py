"""Sobe ao Hub o LTX 2.3 distilled 1.1 W4A8 com TODAS as provas: peso + sidecar, licenca, README,
os quatro MP4 (video + audio) e quatro FLAC da comparacao de 249 quadros (todos os bracos com o
MESMO condicionamento salvo em formato completo), a folha de contato e o JSON das metricas; o
controle de identidade do condicionamento (salvo vs encoder vivo) e a folha do saver quebrado como
negativo; e a rodada de LoRA com gatilho (folha, JSON, os quatro MP4). Recusa se faltar prova.
O W4A4 (controle) NAO sobe como peso -- negativo medido se publica como prova, nao como arquivo que
alguem baixa.

Os nomes de MP4/FLAC vem dos JSONs de corrida (`video[0]`, `audio[0]`), nao de contadores fixos:
os renders do LoRA sem gatilho da fila h ocupam `_00001_` e os novos saem `_00002_`.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot"
ROOT = Path(r"F:\COMFY_PORTABLE")
OUT = ROOT / "ComfyUI" / "output"
CARD = ROOT / "bench" / "hf" / "ltx23-22b-w4a8"
B23 = ROOT / "bench" / "ltx23"
SCR = ROOT / ".scratch"
PESO = Path(r"P:\ComfyBench\checkpoints\ltx-2.3-22b-distilled-1.1_w4a8.safetensors")

# braco -> (json da corrida, nome remoto sem extensao)
CORRIDAS_AV = {
    "bf16_original": "ltx23av_bf16",
    "w4a8_this_file": "ltx23av_w4a8_condf",
    "w4a4_control": "ltx23av_w4a4_condf",
    "gguf_q6k_third_party": "ltx23av_q6k_condf",
}
CORRIDAS_LORA = {
    "no_lora_seed1234": "ltx23trig_base",
    "lora_merged_seed1234": "ltx23trig_merge",
    "lora_bypass_seed1234": "ltx23trig_bypass",
    "no_lora_seed4321": "ltx23trig_seed2",
}


def midia(json_stem: str) -> tuple[Path, Path]:
    j = SCR / f"{json_stem}.json"
    if not j.exists():
        sys.exit(f"falta o JSON da corrida {j}; nao subo nada")
    d = json.loads(j.read_text(encoding="utf-8"))
    if d.get("status") != "success":
        sys.exit(f"corrida {json_stem} nao terminou em success ({d.get('status')}); nao subo nada")
    if not d.get("video") or not d.get("audio"):
        sys.exit(f"corrida {json_stem} sem video/audio no JSON; nao subo nada")
    return OUT / d["video"][0], OUT / d["audio"][0]


provas: dict[str, Path] = {
    "README.md": CARD / "README.md",
    "LICENSE_LTX_2_COMMUNITY.txt": CARD / "LICENSE_LTX_2_COMMUNITY.txt",
    "ltx-2.3-22b-distilled-1.1_w4a8.safetensors": PESO,
    "ltx-2.3-22b-distilled-1.1_w4a8.quant.json": PESO.with_suffix(".quant.json"),
    "av/contato_av.png": B23 / "av" / "contato_av.png",
    "av/comparacao_av.json": B23 / "av" / "comparacao_av.json",
    "conditioning/identity_saved_vs_live_encoder.png": B23 / "cond_identity" / "contato_av.png",
    "conditioning/identity_saved_vs_live_encoder.json": B23 / "cond_identity" / "comparacao_av.json",
    "conditioning/NEGATIVE_ltxv_saver_drops_the_key.png": B23 / "cond_identity_ltxv_saver" / "contato_av.png",
    "conditioning/NEGATIVE_ltxv_saver_drops_the_key.json": B23 / "cond_identity_ltxv_saver" / "comparacao_av.json",
    "lora/contato_av.png": B23 / "lora_trigger" / "contato_av.png",
    "lora/comparacao_av.json": B23 / "lora_trigger" / "comparacao_av.json",
    "lora/merged_vs_bypass.json": B23 / "lora_trigger_par" / "comparacao_av.json",
}
for remoto, stem in CORRIDAS_AV.items():
    mp4, flac = midia(stem)
    provas[f"av/{remoto}.mp4"] = mp4
    provas[f"av/{remoto}.flac"] = flac
for remoto, stem in CORRIDAS_LORA.items():
    mp4, _ = midia(stem)
    provas[f"lora/{remoto}.mp4"] = mp4

faltam = [k for k, v in provas.items() if not v.exists()]
if faltam:
    sys.exit(f"faltam provas, nao subo nada: {faltam}")
if "TBD" in (CARD / "README.md").read_text(encoding="utf-8"):
    sys.exit("README ainda tem TBD; nao subo")

for remoto, local in provas.items():
    print(f"  {remoto:52s} <- {local.name}  ({local.stat().st_size / 2**20:.1f} MiB)")
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
for remoto, local in provas.items():
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remoto, repo_id=REPO, repo_type="model",
                    commit_message=f"LTX 2.3 distilled 1.1 W4A8: {remoto}")
    print(f"  subiu {remoto}  ({local.stat().st_size / 2**20:.1f} MiB)", flush=True)

no_hub = set(api.list_repo_files(REPO, repo_type="model"))
falta_no_hub = [k for k in provas if k not in no_hub]
print("LTX23_UPLOAD_OK" if not falta_no_hub else f"FALTA NO HUB: {falta_no_hub}", flush=True)
print("NAO COBERTO: o peso W4A4 e os renders do LoRA sem gatilho ficam so no git (negativos/registro).")
