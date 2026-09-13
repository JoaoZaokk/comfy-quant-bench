"""Gera video com LTX 2.5 pela API do ComfyUI, um transformer por chamada.

POR QUE PELA API E NAO EM PROCESSO
----------------------------------
O LTX 2.5 nao e um sampler simples: o latente de VIDEO e o de AUDIO sao concatenados antes da
amostragem (`LTXVConcatAVLatent`) e separados depois (`LTXVSeparateAVLatent`), e o guider tem
dois CFG. Reimplementar isso dentro de `quality_ladder.py` seria reescrever nos que ja existem e
ja foram exercitados nesta maquina -- o grafo abaixo e extraido de
`user/default/workflows/LTX25-int8-acceptance-v2.json`, que rodou aqui, com os links lidos do
arquivo em vez de reconstruidos de cabeca.

O EIXO
------
`--transformer` e a UNICA coisa que muda entre bracos. O encoder, o VAE, o prompt, a semente, os
sigmas, a resolucao e o numero de quadros ficam fixos, passados por argumento e registrados no
JSON de saida. Isto e deliberado: uma medicao anterior desta bancada comparou dois bracos que
diferiam em DOIS eixos ao mesmo tempo e a conclusao teve de ser jogada fora.

QUADROS
-------
LTX exige `8n+1`. A 25 fps (o `frame_rate` que o workflow de aceitacao usa), 10 segundos sao 250
quadros, que nao e 8n+1; **249** e o valor legal mais proximo e da 9,96 s. O script recusa
qualquer numero que nao seja 8n+1 em vez de arredondar em silencio.

NAO COBERTO
-----------
Nao mede qualidade: grava o video e o tempo. Nao decodifica o audio (o `audio_vae` entra porque
o latente concatenado exige, mas so o ramo de video e decodificado). Nao verifica que o encoder
casa com o transformer -- passar um gemma de outra versao produz video plausivel e errado.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SIGMAS_DESTILADO = "0.909375, 0.725, 0.421875, 0.0"


def monta_prompt(a) -> dict:
    """O grafo de LTX25-int8-acceptance-v2.json em formato de API."""
    # DisTorch2 distribui os BLOCOS entre placas em vez de descarregar o modelo inteiro para o
    # host. E o que torna o braco BF16 possivel: 39,13 GiB contra 24 GiB de VRAM deu
    # `CUDA error: out of memory` dentro de `mem_get_info` -- a placa esgotou de verdade, nao foi
    # uma alocacao infeliz.
    #
    # `cuda:1` como doadora e instrucao explicita do dono ("offload na 3080ti, ambas as placas sao
    # suas"). O valor NAO e 12 GiB: a 3080 Ti hospeda o embedding do cortex, ~2,3 GiB medidos, que
    # nao pode ser expulso. `--doar-gb` default 6 deixa ~4 GiB de folga sobre o que ja esta la.
    if a.distorch:
        aloc = a.alocacao or f"cuda:1,{a.doar_gb}gb;cpu,*"
        carga_unet = {"class_type": "UNETLoaderDisTorch2MultiGPU",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default",
                                 "compute_device": "cuda:0",
                                 "expert_mode_allocations": aloc,
                                 "eject_models": True}}
        carga_clip = {"class_type": "CLIPLoaderDisTorch2MultiGPU",
                      "inputs": {"clip_name": a.encoder, "type": "ltxv",
                                 "device": "cuda:0",
                                 "expert_mode_allocations": aloc,
                                 "eject_models": True}}
    else:
        carga_unet = {"class_type": "UNETLoader",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default"}}
        carga_clip = {"class_type": "CLIPLoader",
                      "inputs": {"clip_name": a.encoder, "type": "ltxv"}}

    return {
        "1": carga_unet,
        "2": carga_clip,
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": a.audio_vae}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.negative, "clip": ["2", 0]}},
        "7": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": a.size, "height": a.size, "length": a.frames, "batch_size": 1}},
        # `frames_number`, nao `length`. O no de VIDEO ao lado usa `length`, e os dois ficam
        # lado a lado no grafo -- copiar o nome do vizinho e recusado pelo servidor com
        # `required_input_missing`, que e a forma barata de descobrir. Nomes conferidos em
        # /object_info, nao deduzidos.
        "8": {"class_type": "LTXVEmptyLatentAudio",
              "inputs": {"frames_number": a.frames, "frame_rate": a.fps, "batch_size": 1,
                         "audio_vae": ["6", 0]}},
        "9": {"class_type": "LTXVConcatAVLatent",
              "inputs": {"video_latent": ["7", 0], "audio_latent": ["8", 0]}},
        "10": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                          "frame_rate": float(a.fps)}},
        "11": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1],
                          "cfg": a.cfg}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "13": {"class_type": "ManualSigmas", "inputs": {"sigmas": a.sigmas}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0],
                          "sigmas": ["13", 0], "latent_image": ["9", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "VAEDecode",
               "inputs": {"samples": ["16", 0], "vae": ["5", 0]}},
        "18": {"class_type": "SaveAnimatedWEBP" if a.webp else "SaveImage",
               "inputs": ({"images": ["17", 0], "filename_prefix": a.saida,
                           "fps": float(a.fps), "lossless": False, "quality": 90,
                           "method": "default"}
                          if a.webp else
                          {"images": ["17", 0], "filename_prefix": a.saida})},
    }


def posta(base: str, prompt: dict) -> str:
    dados = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(f"{base}/prompt", data=dados,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        corpo = e.read().decode("utf-8", "replace")
        raise SystemExit(f"o servidor RECUSOU o grafo ({e.code}):\n{corpo[:2000]}") from None


def portas_worker(raiz: Path) -> list[str]:
    """As portas que o ComfyUI-MultiGPU escolheu em tempo de execucao, lidas dos logs dele.

    Quando `COMFYUI_MGPU_DISABLED` NAO esta em 1, o pacote sobe um worker por placa em portas
    decididas na hora e o servidor principal encaminha o trabalho para la. O resultado aparece no
    `/history` DO WORKER, e o principal fica com a fila vazia -- o que e indistinguivel de "ja
    terminou" e de "nunca comecou". Isto custou uma corrida aqui: a GPU em 85%, a fila do 8190
    zerada, e o script esperando um id que nunca ia aparecer.
    """
    portas = []
    for log in sorted((raiz / "ComfyUI" / "logs" / "mgpu-workers").glob("gpu-*.log")):
        try:
            texto = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pedaco in texto.split("http://127.0.0.1:")[1:]:
            porta = pedaco.split()[0].split("/")[0].strip(",)")
            if porta.isdigit() and porta not in portas:
                portas.append(porta)
    return portas


def espera(base: str, pid: str, limite_s: int, raiz: Path) -> dict:
    """Espera no /history, e tambem no dos workers do MultiGPU se eles existirem."""
    bases = [base] + [f"http://127.0.0.1:{p}" for p in portas_worker(raiz)]
    if len(bases) > 1:
        print(f"  tambem olhando os workers do MultiGPU: {bases[1:]}", flush=True)
    t0 = time.time()
    ultimo = 0.0
    while time.time() - t0 < limite_s:
        for b in bases:
            try:
                with urllib.request.urlopen(f"{b}/history/{pid}", timeout=60) as r:
                    h = json.load(r)
            except Exception:  # noqa: BLE001 -- worker pode nao estar de pe
                continue
            if pid in h:
                if b != base:
                    print(f"  resultado veio do worker {b}, nao do servidor principal", flush=True)
                return h[pid]
        agora = time.time() - t0
        if agora - ultimo >= 60:
            ultimo = agora
            print(f"  ... {agora / 60:.1f} min", flush=True)
        time.sleep(5)
    raise SystemExit(f"passou de {limite_s}s sem terminar")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transformer", required=True, help="o UNICO eixo que varia entre bracos")
    p.add_argument("--encoder", default="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    p.add_argument("--video-vae", default="ltx-2.5-video-vae-bf16.safetensors")
    p.add_argument("--audio-vae", default="ltx-2.5-audio-vae-bf16.safetensors")
    p.add_argument("--prompt", default="a lone lighthouse on a rocky cliff at dusk, waves "
                                       "breaking against the rocks, the beam sweeping across "
                                       "low clouds, seabirds circling")
    p.add_argument("--negative", default="blurry, out of focus, low contrast, washed out")
    p.add_argument("--frames", type=int, default=249, help="8n+1. 249 = 9,96 s a 25 fps")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--sigmas", default=SIGMAS_DESTILADO)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--saida", default="ltx25")
    p.add_argument("--webp", action="store_true", help="grava um WEBP animado em vez de PNGs")
    p.add_argument("--distorch", action="store_true",
                   help="distribui os blocos entre as placas em vez de descarregar inteiro. "
                        "Obrigatorio para o braco BF16 de 39 GiB, que sem isto da CUDA OOM")
    p.add_argument("--alocacao", default=None,
                   help="string crua de expert_mode_allocations da DisTorch2, ex 'cpu,*'. "
                        "Sobrepoe --doar-gb. Com 249 quadros a doacao cruzada entre placas "
                        "derrubou o processo com access violation; 'cpu,*' evita esse caminho")
    p.add_argument("--doar-gb", type=float, default=6.0,
                   help="quanto da 3080 Ti a DisTorch2 pode usar. NAO e 12: o cortex mora la "
                        "com ~2,3 GiB e nao pode ser expulso")
    p.add_argument("--servidor", default="http://127.0.0.1:8190")
    p.add_argument("--limite", type=int, default=7200)
    p.add_argument("--json", help="grava o registro da corrida aqui")
    a = p.parse_args()

    if (a.frames - 1) % 8 != 0:
        raise SystemExit(f"--frames {a.frames} nao e 8n+1. Os validos ao redor sao "
                         f"{((a.frames - 1) // 8) * 8 + 1} e {((a.frames - 1) // 8 + 1) * 8 + 1}. "
                         "Este script recusa em vez de arredondar sozinho.")

    prompt = monta_prompt(a)
    print(f"transformer : {a.transformer}", flush=True)
    print(f"encoder     : {a.encoder}   (FIXO entre bracos)", flush=True)
    print(f"{a.frames} quadros a {a.fps} fps = {a.frames / a.fps:.2f} s, {a.size}px, "
          f"cfg {a.cfg}, semente {a.seed}", flush=True)

    t0 = time.time()
    pid = posta(a.servidor, prompt)
    print(f"prompt_id {pid}", flush=True)
    saida = espera(a.servidor, pid, a.limite, Path(__file__).resolve().parent.parent)
    segundos = time.time() - t0

    imagens = []
    for no in saida.get("outputs", {}).values():
        imagens += [i.get("filename") for i in no.get("images", []) if i.get("filename")]
    estado = saida.get("status", {})
    print(f"\nterminou em {segundos:.1f}s ({segundos / 60:.1f} min), "
          f"{len(imagens)} arquivo(s) de saida", flush=True)
    if imagens:
        print(f"  primeiro: {imagens[0]}   ultimo: {imagens[-1]}", flush=True)
    if estado.get("status_str") and estado["status_str"] != "success":
        print(f"  STATUS {estado['status_str']}", flush=True)

    reg = {"transformer": a.transformer, "encoder": a.encoder, "frames": a.frames,
           "fps": a.fps, "size": a.size, "cfg": a.cfg, "sigmas": a.sigmas, "seed": a.seed,
           "prompt": a.prompt, "segundos": segundos, "arquivos": imagens,
           "s_por_quadro": segundos / a.frames,
           "distorch": a.distorch, "doar_gb": a.doar_gb if a.distorch else None}
    if a.json:
        Path(a.json).write_text(json.dumps(reg, indent=2), encoding="utf-8")
        print(f"registro em {a.json}", flush=True)

    print("\nNAO COBERTO: nada aqui julga a qualidade do video, so que ele saiu e quanto "
          "demorou. O audio nao e decodificado. Nao confere que o encoder casa com o "
          "transformer -- gemma de outra versao produz video plausivel e errado.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
