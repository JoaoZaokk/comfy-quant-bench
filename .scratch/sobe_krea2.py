"""Sobe ao Hub os dois builds do Krea 2 Turbo (W4A4 e misto) com as provas, cumprindo a Krea 2
Community License §3.1-3.3: nome do repo comeca com 'Krea', copia do acordo (pdf + txt), NOTICE.txt
com a frase prescrita e a declaracao de modificacao, e o repo GATED (cada destinatario aceita o
acordo, como no repo de origem). O token e o do ambiente; este script nao cria nem grava token.
Recusa se faltar prova ou se o README tiver TBD.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "JoaoZaokk/Krea-2-Turbo-W4A4-ConvRot"
ROOT = Path(r"F:\COMFY_PORTABLE")
CARD = ROOT / "bench" / "hf" / "krea2-turbo-quant"
DM = ROOT / "ComfyUI" / "models" / "diffusion_models"

provas: dict[str, Path] = {
    "README.md": CARD / "README.md",
    "NOTICE.txt": CARD / "NOTICE.txt",
    "LICENSE_KREA_2_COMMUNITY.pdf": CARD / "LICENSE_KREA_2_COMMUNITY.pdf",
    "LICENSE_KREA_2_COMMUNITY.txt": CARD / "LICENSE_KREA_2_COMMUNITY.txt",
    "krea2_turbo_w4a4.safetensors": DM / "krea2_turbo_w4a4.safetensors",
    "krea2_turbo_w4a4.quant.json": DM / "krea2_turbo_w4a4.quant.json",
    "krea2_turbo_mixed.safetensors": DM / "krea2_turbo_mixed.safetensors",
    "krea2_turbo_mixed.quant.json": DM / "krea2_turbo_mixed.quant.json",
}
for img in sorted((CARD / "images").glob("*.png")):
    provas[f"images/{img.name}"] = img

faltam = [k for k, v in provas.items() if not v.exists()]
if faltam:
    sys.exit(f"faltam provas, nao subo nada: {faltam}")
readme = (CARD / "README.md").read_text(encoding="utf-8")
if "TBD" in readme:
    sys.exit("README ainda tem TBD; nao subo")
for must in ("gated: auto", "NOTICE.txt", "LICENSE_KREA_2_COMMUNITY.pdf", "Krea 2 Community License"):
    if must not in readme:
        sys.exit(f"README sem '{must}'; a licenca exige; nao subo")
if not REPO.split("/")[1].startswith("Krea"):
    sys.exit("nome do repo deve comecar com 'Krea' (licenca 3.1(b))")

for remoto, local in provas.items():
    print(f"  {remoto:40s} <- {local.name}  ({local.stat().st_size / 2**20:.1f} MiB)")
api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
for remoto, local in provas.items():
    api.upload_file(path_or_fileobj=str(local), path_in_repo=remoto, repo_id=REPO, repo_type="model",
                    commit_message=f"Krea 2 Turbo ConvRot derivatives: {remoto}")
    print(f"  subiu {remoto}  ({local.stat().st_size / 2**20:.1f} MiB)", flush=True)
try:
    api.update_repo_settings(repo_id=REPO, gated="auto")
    print("gated: auto (cada destinatario aceita o acordo)")
except Exception as e:  # noqa: BLE001 -- o upload ja esta feito; o gating e reportado, nao presumido
    print("AVISO: nao consegui ligar o gating pela API:", str(e)[:200])

no_hub = set(api.list_repo_files(REPO, repo_type="model"))
falta_no_hub = [k for k in provas if k not in no_hub]
info = api.model_info(REPO)
print("gated no hub:", getattr(info, "gated", None))
print("KREA2_UPLOAD_OK" if not falta_no_hub else f"FALTA NO HUB: {falta_no_hub}", flush=True)
print("NAO COBERTO: os builds cg16/cg64 do teto nao existem mais no disco e nao sobem; o encoder qwen3vl_4b W4A8 da suite fica em outro repo (nao subiu aqui).")
