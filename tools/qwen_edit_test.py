"""Roda o Qwen-Image-Edit fazendo EDICAO, que e o caminho que da nome ao modelo.

POR QUE ISTO EXISTE
-------------------
Em 2026-09-12 esta bancada mediu o Qwen-Image-Edit 2511 com `quality_ladder.py`, que amostra do
ruido: seis prompts, nenhuma imagem de entrada. Isso e **text-to-image**. O modelo e um EDITOR, e
o veredito publicado ("W4A8 funciona") descrevia um caminho que nao e o dele. O dono apontou o
erro. Este arquivo e o conserto.

O que muda no grafo, e por que nao da para reaproveitar o ladder:

    LoadImage -> ImageScaleToTotalPixels -> TextEncodeQwenImageEditPlus (com vae e image1)
                                         -> VAEEncode -> latent_image do KSampler
    as duas condicoes passam por FluxKontextMultiReferenceLatentMethod(index_timestep_zero)
    o modelo passa por ModelSamplingAuraFlow(shift) e CFGNorm

Nada disso existe no ladder, que monta um latente de zeros e amostra. O grafo abaixo e lido de
`user/default/workflows/Qwen_Edit_2511_DualGPU_3090_3080Ti_FIXED.json`, que roda nesta maquina,
com os links extraidos do arquivo -- trocando so o carregador Nunchaku por um `UNETLoader`
comum, porque o eixo aqui e o FORMATO do transformer.

O EIXO
------
`--transformer` e a unica coisa que varia. Encoder, VAE, imagem de entrada, instrucao, semente,
passos, cfg, sampler e escala ficam fixos e vao gravados no JSON.

NAO COBERTO
-----------
Nao julga a edicao: grava a imagem e o tempo. Nao mede se a instrucao foi obedecida -- isso
precisa de alguem olhando, ou de uma metrica que esta bancada nao tem. Uma unica imagem de
entrada por corrida (o no aceita ate tres; usar duas mede outra coisa).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def monta_prompt(a) -> dict:
    # O braco BF16 (38,05 GiB) nao cabe por este caminho nem com a placa limpa: aqui ele coexiste
    # com o encoder de 15,45 GiB e ainda faz um VAEEncode, e o resultado e
    # `CUDA error: out of memory` vindo de `mem_get_info`. O ladder de t2i roda o mesmo arquivo a
    # 113,9 s/render porque ele mesmo forca NORMAL_VRAM antes de carregar; o servidor nao faz isso.
    #
    # DisTorch2 muda ONDE os blocos moram, nao a matematica -- entao a imagem comparada continua
    # valida. O TEMPO nao continua: um braco espalhado paga transferencia por passo e o s/render
    # dele nao se compara com o de um braco residente. Por isso o JSON grava `distorch` por corrida.
    if a.distorch:
        carga_unet = {"class_type": "UNETLoaderDisTorch2MultiGPU",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default",
                                 "compute_device": "cuda:0",
                                 "expert_mode_allocations": a.alocacao,
                                 "eject_models": True}}
    else:
        carga_unet = {"class_type": "UNETLoader",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default"}}

    return {
        "1": carga_unet,
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.encoder, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": a.imagem}},
        # 1 megapixel e o que o workflow de fabrica usa. Sem isto uma entrada grande muda o
        # custo por braco e a comparacao deixa de ser casada.
        # `resolution_steps` nao tem default aplicado pela API: o servidor recusa o grafo com
        # `required_input_missing` se ele faltar, mesmo o /object_info anunciando default 1.
        # Conferido em /object_info depois da recusa; o valor 1 e o que o workflow de fabrica usa.
        "5": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                         "megapixels": 1.0, "resolution_steps": 1}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "image1": ["5", 0],
                         "prompt": a.instrucao}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "image1": ["5", 0],
                         "prompt": a.negativa}},
        "8": {"class_type": "FluxKontextMultiReferenceLatentMethod",
              "inputs": {"conditioning": ["6", 0],
                         "reference_latents_method": "index_timestep_zero"}},
        "9": {"class_type": "FluxKontextMultiReferenceLatentMethod",
              "inputs": {"conditioning": ["7", 0],
                         "reference_latents_method": "index_timestep_zero"}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "11": {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": ["1", 0], "shift": a.shift}},
        "12": {"class_type": "CFGNorm", "inputs": {"model": ["11", 0], "strength": 1.0}},
        "13": {"class_type": "KSampler",
               "inputs": {"model": ["12", 0], "positive": ["8", 0], "negative": ["9", 0],
                          "latent_image": ["10", 0], "seed": a.seed, "steps": a.steps,
                          "cfg": a.cfg, "sampler_name": a.sampler, "scheduler": a.scheduler,
                          "denoise": a.denoise}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage",
               "inputs": {"images": ["14", 0], "filename_prefix": a.saida}},
    }


def portas_worker(raiz: Path) -> list[str]:
    """Ver tools/ltx25_video.py: com os workers do MultiGPU ligados o resultado cai no /history
    DELES e o do servidor principal fica vazio, o que e indistinguivel de 'nunca comecou'."""
    portas = []
    d = raiz / "ComfyUI" / "logs" / "mgpu-workers"
    for log in sorted(d.glob("gpu-*.log")) if d.is_dir() else []:
        try:
            texto = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pedaco in texto.split("http://127.0.0.1:")[1:]:
            porta = pedaco.split()[0].split("/")[0].strip(",)")
            if porta.isdigit() and porta not in portas:
                portas.append(porta)
    return portas


def roda(a, base: str, raiz: Path) -> dict:
    dados = json.dumps({"prompt": monta_prompt(a)}).encode()
    req = urllib.request.Request(f"{base}/prompt", data=dados,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            pid = json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"o servidor RECUSOU o grafo ({e.code}):\n"
                         f"{e.read().decode('utf-8', 'replace')[:1800]}") from None

    bases = [base] + [f"http://127.0.0.1:{p}" for p in portas_worker(raiz)]
    t0 = time.time()
    while time.time() - t0 < a.limite:
        for b in bases:
            try:
                with urllib.request.urlopen(f"{b}/history/{pid}", timeout=60) as r:
                    h = json.load(r)
            except Exception:  # noqa: BLE001
                continue
            if pid in h:
                return {"saida": h[pid], "segundos": time.time() - t0, "onde": b}
        time.sleep(4)
    raise SystemExit(f"passou de {a.limite}s sem terminar")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transformer", required=True, help="o UNICO eixo que varia")
    p.add_argument("--encoder", default="qwen_2.5_vl_7b.safetensors")
    p.add_argument("--vae", default="qwen_image_vae.safetensors")
    p.add_argument("--imagem", required=True, help="nome dentro de ComfyUI/input")
    p.add_argument("--instrucao", required=True)
    p.add_argument("--negativa", default=" ")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=2.5)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--denoise", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--saida", default="qwen_edit")
    p.add_argument("--servidor", default="http://127.0.0.1:8190")
    p.add_argument("--limite", type=int, default=1800)
    p.add_argument("--distorch", action="store_true",
                   help="espalha os blocos em vez de carregar inteiro. Obrigatorio para o braco "
                        "BF16 de 38 GiB, que sem isto da CUDA OOM neste grafo")
    p.add_argument("--alocacao", default="cpu,40gb",
                   help="expert_mode_allocations da DisTorch2. A unidade de byte e obrigatoria: "
                        "o parser so aceita o curinga '*' no ramo que ve g/m/k/b na string")
    p.add_argument("--json", help="grava o registro da corrida aqui")
    a = p.parse_args()

    print(f"transformer : {a.transformer}", flush=True)
    print(f"entrada     : {a.imagem}", flush=True)
    print(f"instrucao   : {a.instrucao[:90]}", flush=True)
    r = roda(a, a.servidor, Path(__file__).resolve().parent.parent)
    imagens = [i.get("filename")
               for no in r["saida"].get("outputs", {}).values()
               for i in no.get("images", [])]
    st = r["saida"].get("status", {}).get("status_str")
    print(f"  {r['segundos']:.1f}s, status {st}, {len(imagens)} saida(s): {imagens[:2]}", flush=True)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"transformer": a.transformer, "encoder": a.encoder, "imagem": a.imagem,
             "instrucao": a.instrucao, "seed": a.seed, "steps": a.steps, "cfg": a.cfg,
             "shift": a.shift, "segundos": r["segundos"], "arquivos": imagens,
             "distorch": a.distorch, "alocacao": a.alocacao if a.distorch else None,
             "status": st}, indent=2), encoding="utf-8")
    return 0 if st == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
