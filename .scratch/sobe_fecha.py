"""Uploads do fechamento 2026-09-14, um item por chamada (`--item`), cada um recusando sem a prova:

  ltx25int8  -> repo LTX-2.5 existente: os dois int8 nossos + sidecars + folha/JSON de bench/ltx25/int8_ours
  heretic    -> repo Gemma heretic existente: w4a4_convrot + w4a4_smooth + sidecars + JSON/folhas de encoder_heretic
  capybara   -> repo Hunyuan existente: capybara_v0.1_w4a8r + sidecar + ladder de bench/capybara_w4a8
  gemma      -> repo NOVO Gemma-3-12B-it-W4A8-ConvRot: gemma_3_12B_it_w4a8 + sidecar + card + provas de encoder_w4a8
  readmes    -> re-sobe so os READMEs dos repos tocados (2.5, heretic, hunyuan, 2.3)

Token do ambiente. Nao cria token. Imprime <ITEM>_UPLOAD_OK ou o que falta.
"""
import argparse
import os
import sys
from pathlib import Path

# O token de ESCRITA mora em <repo>/.hf/token. Fixar aqui porque HfApi() sem token pega o que o
# shell tiver, e F:/hf-cache/token e de LEITURA: o sintoma e 403 "you must use a write token"
# depois de o upload ja ter comecado, e os dois arquivos tem 37 bytes, entao tamanho nao distingue.
# Medido em 2026-09-20, apos tomar exatamente esse 403.
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[1] / ".hf")
for _k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    os.environ.pop(_k, None)

from huggingface_hub import HfApi

ROOT = Path(r"F:\COMFY_PORTABLE")
B = ROOT / "bench"
HF = B / "hf"
TE = ROOT / "ComfyUI" / "models" / "text_encoders"
D25 = Path(r"D:/ComfyUI-Models/diffusion_models")
OUT = ROOT / "ComfyUI" / "output"

ITENS = {
    "ltx25int8": ("JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot", {
        "README.md": HF / "ltx25-22b-w4a8" / "README.md",
        "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8.safetensors": D25 / "ltx-2.5-22b-distilled-transformer-bf16_int8.safetensors",
        "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8.quant.json": D25 / "ltx-2.5-22b-distilled-transformer-bf16_int8.quant.json",
        "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors": D25 / "ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.safetensors",
        "int8_ours/ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.quant.json": D25 / "ltx-2.5-22b-distilled-transformer-bf16_int8_convrot.quant.json",
        "int8_ours/contato_av.png": B / "ltx25" / "int8_ours" / "contato_av.png",
        "int8_ours/comparacao_av.json": B / "ltx25" / "int8_ours" / "comparacao_av.json",
        "int8_ours/int8_no_rotation.mp4": OUT / "ltx25av_ours_int8_av_00001_.mp4",
        "int8_ours/int8_no_rotation.flac": OUT / "ltx25av_ours_int8_audio_00001.flac",
        "int8_ours/int8_convrot.mp4": OUT / "ltx25av_ours_int8_convrot_av_00001_.mp4",
        "int8_ours/int8_convrot.flac": OUT / "ltx25av_ours_int8_convrot_audio_00001.flac",
    }),
    "heretic": ("JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8", {
        "README.md": HF / "gemma3-12b-heretic-w4a8" / "README.md",
        # os PESOS W4A4 nao sobem: no caminho destravado mudam a cena (crepusculo -> dia), negativo
        # medido se publica como prova, nao como checkpoint (bench/ltx23/encoder_heretic). Sobem os
        # sidecars, para que o formato exato do que foi medido fique no registro.
        # 05:10 -- o dono mandou subir os pesos; vao ao lado da medicao, rotulados no card.
        "w4a4/gemma_3_12B_it_heretic_w4a4_convrot.safetensors": TE / "gemma_3_12B_it_heretic_w4a4_convrot.safetensors",
        "w4a4/gemma_3_12B_it_heretic_w4a4_convrot.quant.json": TE / "gemma_3_12B_it_heretic_w4a4_convrot.quant.json",
        "w4a4/gemma_3_12B_it_heretic_w4a4_smooth.safetensors": TE / "gemma_3_12B_it_heretic_w4a4_smooth.safetensors",
        "w4a4/gemma_3_12B_it_heretic_w4a4_smooth.quant.json": TE / "gemma_3_12B_it_heretic_w4a4_smooth.quant.json",
        "ltx23_encoder_test/locked_render_contato_av.png": B / "ltx23" / "encoder_heretic_locked" / "contato_av.png",
        "ltx23_encoder_test/locked_render_comparacao_av.json": B / "ltx23" / "encoder_heretic_locked" / "comparacao_av.json",
        "ltx23_encoder_test/conditioning_vs_bf16.json": B / "ltx23" / "encoder_cond_heretic.json",
        "ltx23_encoder_test/render_contato_av.png": B / "ltx23" / "encoder_heretic" / "contato_av.png",
        "ltx23_encoder_test/render_comparacao_av.json": B / "ltx23" / "encoder_heretic" / "comparacao_av.json",
        "ltx23_encoder_test/heretic_vs_factory_contato_av.png": B / "ltx23" / "encoder_heretic_vs_factory" / "contato_av.png",
        "ltx23_encoder_test/heretic_vs_factory_comparacao_av.json": B / "ltx23" / "encoder_heretic_vs_factory" / "comparacao_av.json",
    }),
    "capybara": ("JoaoZaokk/HunyuanVideo-1.5-720p-T2V-Quantized", {
        "README.md": HF / "hunyuan15-quant" / "README.md",
        "capybara_v0.1_w4a8r.safetensors": Path(r"P:/ComfyBench/diffusion_models/capybara_v0.1_w4a8r.safetensors"),
        "capybara_v0.1_w4a8r.quant.json": Path(r"P:/ComfyBench/diffusion_models/capybara_v0.1_w4a8r.quant.json"),
        "images/capybara_w4a8_ladder.json": B / "capybara_w4a8" / "ladder.json",
    }),
    "gemma": ("JoaoZaokk/Gemma-3-12B-it-W4A8-ConvRot", {
        "README.md": HF / "gemma3-12b-w4a8" / "README.md",
        "gemma_3_12B_it_w4a8.safetensors": Path(r"P:/ComfyBench/text_encoders/gemma_3_12B_it_w4a8.safetensors"),
        "gemma_3_12B_it_w4a8.quant.json": Path(r"P:/ComfyBench/text_encoders/gemma_3_12B_it_w4a8.quant.json"),
        "ltx23_encoder_test/conditioning_vs_bf16.json": B / "ltx23" / "encoder_cond_factory.json",
        "ltx23_encoder_test/render_contato_av.png": B / "ltx23" / "encoder_w4a8" / "contato_av.png",
        "ltx23_encoder_test/render_comparacao_av.json": B / "ltx23" / "encoder_w4a8" / "comparacao_av.json",
    }),
}
READMES = {
    "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot": HF / "ltx25-22b-w4a8" / "README.md",
    "JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8": HF / "gemma3-12b-heretic-w4a8" / "README.md",
    "JoaoZaokk/HunyuanVideo-1.5-720p-T2V-Quantized": HF / "hunyuan15-quant" / "README.md",
    "JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot": HF / "ltx23-22b-w4a8" / "README.md",
}


def sobe(api: HfApi, repo: str, provas: dict, extra_imgs: Path | None = None) -> bool:
    if extra_imgs and extra_imgs.exists():
        for img in sorted(extra_imgs.glob("*.png")):
            provas[f"images/{img.name}"] = img
    faltam = [k for k, v in provas.items() if not v.exists()]
    if faltam:
        print(f"faltam provas, nao subo {repo}: {faltam}")
        return False
    for k, v in provas.items():
        if k == "README.md" and "TBD" in v.read_text(encoding="utf-8"):
            print(f"README de {repo} com TBD; nao subo")
            return False
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    for velho in [f for f in api.list_repo_files(repo, repo_type="model") if f.startswith("w4a4_not_published/")]:
        api.delete_file(velho, repo_id=repo, repo_type="model", commit_message=f"closing round 2026-09-14: {velho} moved beside the weights in w4a4/")
        print(f"  apagou {velho}", flush=True)
    for remoto, local in provas.items():
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remoto, repo_id=repo, repo_type="model",
                        commit_message=f"closing round 2026-09-14: {remoto}")
        print(f"  subiu {remoto}  ({local.stat().st_size / 2**20:.1f} MiB)", flush=True)
    no_hub = set(api.list_repo_files(repo, repo_type="model"))
    falta = [k for k in provas if k not in no_hub]
    if falta:
        print(f"FALTA NO HUB {repo}: {falta}")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--item", required=True, choices=[*ITENS, "readmes"])
    p.add_argument("--sem-readme", action="store_true", help="sobe pesos e provas agora; o README vai depois via --item readmes")
    a = p.parse_args()
    api = HfApi()
    if a.item == "readmes":
        ok = True
        for repo, readme in READMES.items():
            ok &= sobe(api, repo, {"README.md": readme})
        print("READMES_UPLOAD_OK" if ok else "READMES_FALHOU")
        return 0 if ok else 1
    repo, provas = ITENS[a.item]
    extra = (B / "capybara_w4a8") if a.item == "capybara" else None
    provas = dict(provas)
    if a.sem_readme:
        provas.pop("README.md")
    ok = sobe(api, repo, provas, extra)
    print(f"{a.item.upper()}_UPLOAD_OK" if ok else f"{a.item.upper()}_FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
