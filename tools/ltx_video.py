"""Gera VIDEO E AUDIO com LTX 2.3 / 2.5 pela API do ComfyUI, um transformer por chamada.

POR QUE PELA API E NAO EM PROCESSO
----------------------------------
O LTX 2.x nao e um sampler simples: o latente de VIDEO e o de AUDIO sao concatenados antes da
amostragem (`LTXVConcatAVLatent`) e separados depois (`LTXVSeparateAVLatent`), e o guider tem
dois CFG. Reimplementar isso dentro de `quality_ladder.py` seria reescrever nos que ja existem e
ja foram exercitados nesta maquina -- o grafo abaixo e extraido de
`user/default/workflows/LTX25-int8-acceptance-v2.json`, que rodou aqui, com os links lidos do
arquivo em vez de reconstruidos de cabeca, e do template oficial `video_ltx2_3_t2v.json` para a
metade 2.3 (loader de checkpoint unico, `LTXVAudioVAELoader`, `LTXAVTextEncoderLoader`).

O AUDIO, E POR QUE A VERSAO ANTERIOR DESTE ARQUIVO ESTAVA ERRADA
----------------------------------------------------------------
`tools/ltx25_video.py` decodificava so o ramo de video e gravava PNGs. O dono apontou o erro em
2026-09-13: *"um gerador de video como o LTX nao gera somente imagens, ele gera imagem e audio ao
mesmo tempo, entao voce tem que comparar os dois"*. Toda comparacao publicada ate entao (o card
do LTX 2.5 no Hub) media metade do que o modelo produz. Este arquivo decodifica os DOIS ramos
(`LTXVAudioVAEDecode` sobre a segunda saida do `LTXVSeparateAVLatent`) e grava tres coisas por
corrida:

  PNGs   um por quadro, sem perda -- e o que `compara_av.py` mede
  FLAC   o audio sem perda -- idem
  MP4    quadros + audio juntos, para uma pessoa assistir; NAO e usado em metrica (h264 e lossy)

O EIXO
------
O transformer e a UNICA coisa que muda entre bracos. O encoder, os VAEs, o prompt, a semente, os
sigmas, a resolucao e o numero de quadros ficam fixos, passados por argumento e registrados no
JSON de saida. Isto e deliberado: uma medicao anterior desta bancada comparou dois bracos que
diferiam em DOIS eixos ao mesmo tempo e a conclusao teve de ser jogada fora.

Um LoRA (`--lora`) e um segundo eixo, so para o teste de LoRA: usa-lo com o mesmo transformer com
e sem e o desenho certo; troca-lo junto com o transformer nao e.

DUAS VERSOES, DOIS LAYOUTS DE ARQUIVO
-------------------------------------
LTX 2.5 e distribuido em pecas (`diffusion_models/`, `text_encoders/gemma4-with-proj`, dois VAEs)
e carrega por `UNETLoader` + `CLIPLoader(ltxv)` + `VAELoader` x2. LTX 2.3 e UM checkpoint
(DiT + VAE de video + VAE de audio + vocoder + projecao de texto) e carrega por
`CheckpointLoaderSimple` (modelo e VAE), `LTXVAudioVAELoader` (mesmo arquivo) e
`LTXAVTextEncoderLoader` (gemma 3 12B de `text_encoders/` + a projecao lida do mesmo checkpoint).
`--checkpoint` liga o segundo modo. Nele, `--transformer` ou `--gguf` trocam SO o modelo,
mantendo VAEs e projecao do checkpoint -- e assim que um braco quantizado e um GGUF de terceiro
se comparam ao BF16 sem mudar mais nada.

QUADROS
-------
LTX exige `8n+1`. A 25 fps, 10 segundos sao 250 quadros, que nao e 8n+1; **249** e o valor legal
mais proximo e da 9,96 s. O script recusa qualquer numero que nao seja 8n+1 em vez de arredondar.

NAO COBERTO
-----------
Nao mede qualidade: grava video, audio e tempo. Nao verifica que o encoder casa com o transformer
-- gemma de outra versao produz video plausivel e errado. Nao confere que o LoRA casou com o
modelo: o ComfyUI loga `NOT LOADED` por chave que nao casa e este script nao le o log do
servidor; `probe_lora_requant.py` e quem conta isso.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# Lightricks, README do Lightricks/LTX-2.5: destilado, 3 passos, cfg 1.
SIGMAS_DESTILADO_25 = "0.909375, 0.725, 0.421875, 0.0"
# Lightricks, README do Lightricks/LTX-2.3: "distilled v1.1, 8 steps, CFG=1". A lista e a do
# primeiro estagio do template oficial `video_ltx2_3_t2v.json` (no 252, ManualSigmas).
SIGMAS_DESTILADO_23 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def monta_prompt(a) -> dict:
    """O grafo em formato de API. Devolve o dict pronto para POST /prompt."""
    # DisTorch2 distribui os BLOCOS entre placas em vez de descarregar o modelo inteiro para o
    # host. E o que torna o braco BF16 possivel: 39,13 GiB contra 24 GiB de VRAM deu
    # `CUDA error: out of memory` dentro de `mem_get_info` -- a placa esgotou de verdade.
    #
    # `cuda:1` como doadora e instrucao explicita do dono ("offload na 3080ti, ambas as placas sao
    # suas"). O valor NAO e 12 GiB: a 3080 Ti hospeda o embedding do cortex, ~2,3 GiB medidos, que
    # nao pode ser expulso. `--doar-gb` default 6 deixa ~4 GiB de folga sobre o que ja esta la.
    aloc = a.alocacao or f"cuda:1,{a.doar_gb}gb;cpu,*"
    g: dict = {}

    if a.checkpoint:
        # ---- LTX 2.3: checkpoint unico ----
        # Cada loader auxiliar le o arquivo INTEIRO que recebe (`load_torch_file`), entao apontar
        # `LTXVAudioVAELoader` e `LTXAVTextEncoderLoader` para o checkpoint de 43 GiB mapeia 43 GiB
        # duas vezes a mais por braco -- e foi um mmap de 39 GiB lido do SMB que derrubou o servidor
        # no 2.5. `--audio-checkpoint` e `--proj-checkpoint` apontam para arquivos pequenos em
        # `checkpoints/` com SO o VAE de audio + vocoder (1329 tensores) e SO a projecao (4).
        # Conferido byte a byte em 2026-09-13: os 1503 tensores preservados sao identicos entre o
        # BF16, o W4A8 e o W4A4, e os arquivos soltos sao identicos ao checkpoint. Default: o
        # proprio --checkpoint. O VAE de VIDEO sai do mesmo loader do modelo (saida 2).
        audio_ck = a.audio_checkpoint or a.checkpoint
        proj_ck = a.proj_checkpoint or a.checkpoint
        # O no 1c so existe se algo precisar do checkpoint: o MODELO (sem --transformer/--gguf)
        # ou o VAE de video (sem --video-vae). Um checkpoint unico de 43 GiB carregado pelo
        # caminho de checkpoint + DisTorch2 derrubou o servidor duas vezes (access violation no
        # mmap, `torch/storage.py __getitem__`); o transformer sozinho pelo UNETLoader nao.
        precisa_1c = not (a.transformer or a.gguf) or not a.video_vae
        if precisa_1c and a.distorch and not (a.transformer or a.gguf):
            g["1c"] = {"class_type": "CheckpointLoaderSimpleDisTorch2MultiGPU",
                       "inputs": {"ckpt_name": a.checkpoint, "compute_device": "cuda:0",
                                  "expert_mode_allocations": aloc, "eject_models": True}}
        elif precisa_1c:
            g["1c"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.checkpoint}}
        if a.transformer:
            g["1"] = ({"class_type": "UNETLoaderDisTorch2MultiGPU",
                       "inputs": {"unet_name": a.transformer, "weight_dtype": "default",
                                  "compute_device": "cuda:0", "expert_mode_allocations": aloc,
                                  "eject_models": True}}
                      if a.distorch else
                      {"class_type": "UNETLoader",
                       "inputs": {"unet_name": a.transformer, "weight_dtype": "default"}})
            modelo = ["1", 0]
        elif a.gguf:
            g["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": a.gguf}}
            modelo = ["1", 0]
        else:
            modelo = ["1c", 0]
        g["2"] = {"class_type": "LTXAVTextEncoderLoader",
                  "inputs": {"text_encoder": a.encoder, "ckpt_name": proj_ck, "device": "default"}}
        if a.video_vae:
            g["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}}
            vae_video = ["5", 0]
        else:
            vae_video = ["1c", 2]
        g["6"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": audio_ck}}
    else:
        # ---- LTX 2.5: pecas separadas ----
        if a.distorch:
            g["1"] = {"class_type": "UNETLoaderDisTorch2MultiGPU",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default",
                                 "compute_device": "cuda:0", "expert_mode_allocations": aloc,
                                 "eject_models": True}}
            g["2"] = {"class_type": "CLIPLoaderDisTorch2MultiGPU",
                      "inputs": {"clip_name": a.encoder, "type": "ltxv", "device": "cuda:0",
                                 "expert_mode_allocations": aloc, "eject_models": True}}
        else:
            g["1"] = {"class_type": "UNETLoader",
                      "inputs": {"unet_name": a.transformer, "weight_dtype": "default"}}
            g["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": a.encoder, "type": "ltxv"}}
        modelo = ["1", 0]
        g["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}}
        vae_video = ["5", 0]
        g["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.audio_vae}}

    if a.lora:
        # Fusao (`LoraLoaderModelOnly`): sobre peso quantizado o ComfyUI dequantiza, soma o delta
        # e REQUANTIZA para 4 bits (`comfy/ops.py:1455`, `set_weight` ->
        # `requantize_from_float`). Bypass (`LoraLoaderBypassModelOnly`): o delta fica como ramo
        # BF16 de baixo posto no forward e o peso nao e tocado. Os dois sao eixos legitimos.
        g["1L"] = {"class_type": "LoraLoaderBypassModelOnly" if a.lora_bypass else "LoraLoaderModelOnly",
                   "inputs": {"model": modelo, "lora_name": a.lora,
                              "strength_model": a.lora_strength}}
        modelo = ["1L", 0]

    g.update({
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["2", 0]}},
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
               "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": float(a.fps)}},
        "11": {"class_type": "CFGGuider",
               "inputs": {"model": modelo, "positive": ["10", 0], "negative": ["10", 1],
                          "cfg": a.cfg}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "13": {"class_type": "ManualSigmas", "inputs": {"sigmas": a.sigmas}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "15": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0],
                          "sigmas": ["13", 0], "latent_image": ["9", 0]}},
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": vae_video}},
        # A segunda saida do Separate e o latente de AUDIO. Decodifica-lo custa segundos; nao
        # decodifica-lo custou uma comparacao publicada pela metade.
        "19": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["16", 1], "audio_vae": ["6", 0]}},
        "18": {"class_type": "SaveImage",
               "inputs": {"images": ["17", 0], "filename_prefix": a.saida}},
        # DynamicCombo na API: a chave escolhida vai no proprio id do input (`format: flac`);
        # sub-inputs iriam como `format.quality`. Lido em comfy_api/latest/_io.py:1219-1270.
        "20": {"class_type": "SaveAudioAdvanced",
               "inputs": {"audio": ["19", 0], "filename_prefix": a.saida + "_audio",
                          "format": "flac"}},
        "21": {"class_type": "CreateVideo",
               "inputs": {"images": ["17", 0], "fps": float(a.fps), "audio": ["19", 0]}},
        "22": {"class_type": "SaveVideo",
               "inputs": {"video": ["21", 0], "filename_prefix": a.saida + "_av",
                          "format": "mp4", "codec": "auto"}},
    })
    return g


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


def arquivos_de_saida(saida: dict) -> dict[str, list[str]]:
    """Separa o que cada no de saida gravou: quadros (SaveImage), audio, video (mp4)."""
    quadros, audio, video = [], [], []
    for no_id, no in saida.get("outputs", {}).items():
        for chave, itens in no.items():
            if not isinstance(itens, list):
                continue
            for it in itens:
                if not isinstance(it, dict) or not it.get("filename"):
                    continue
                nome = it["filename"]
                if it.get("subfolder"):
                    nome = it["subfolder"].replace("\\", "/") + "/" + nome
                if nome.lower().endswith(".png"):
                    quadros.append(nome)
                elif nome.lower().endswith((".flac", ".wav", ".mp3", ".opus")):
                    audio.append(nome)
                elif nome.lower().endswith((".mp4", ".webm", ".mkv")):
                    video.append(nome)
    return {"quadros": quadros, "audio": audio, "video": video}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transformer", default=None,
                   help="arquivo em diffusion_models/. Em modo 2.5 e obrigatorio; em modo "
                        "--checkpoint troca so o MODELO, mantendo VAEs e projecao do checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="LTX 2.3: checkpoint UNICO em checkpoints/ (DiT + VAEs + projecao)")
    p.add_argument("--audio-checkpoint", default=None,
                   help="com --checkpoint: arquivo em checkpoints/ de onde LTXVAudioVAELoader le o VAE "
                        "de audio + vocoder (default: o proprio --checkpoint; um arquivo pequeno so com "
                        "audio_vae.*/vocoder.* evita mapear o checkpoint inteiro de novo)")
    p.add_argument("--proj-checkpoint", default=None,
                   help="com --checkpoint: arquivo em checkpoints/ de onde LTXAVTextEncoderLoader le a "
                        "projecao de texto (default: o proprio --checkpoint)")
    p.add_argument("--gguf", default=None,
                   help="com --checkpoint: modelo de um GGUF de terceiro (UnetLoaderGGUF)")
    p.add_argument("--encoder", default="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
                   help="2.5: gemma4-with-proj; 2.3: um gemma 3 12B de text_encoders/")
    p.add_argument("--video-vae", default=None,
                   help="2.5: obrigatorio (default ltx-2.5-video-vae-bf16); 2.3: opcional, "
                        "sobrepoe o VAE do checkpoint")
    p.add_argument("--audio-vae", default="ltx-2.5-audio-vae-bf16.safetensors",
                   help="so no modo 2.5; no 2.3 vem do checkpoint")
    p.add_argument("--lora", default=None, help="arquivo em loras/ (segundo eixo, so para teste de LoRA)")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--lora-bypass", action="store_true",
                   help="LoraLoaderBypassModelOnly: delta como ramo BF16, peso intocado")
    p.add_argument("--prompt", default="a lone lighthouse on a rocky cliff at dusk, waves "
                                       "breaking against the rocks, the beam sweeping across "
                                       "low clouds, seabirds circling")
    p.add_argument("--negative", default="blurry, out of focus, low contrast, washed out")
    p.add_argument("--frames", type=int, default=249, help="8n+1. 249 = 9,96 s a 25 fps")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--sigmas", default=None,
                   help=f"default: 2.5 -> '{SIGMAS_DESTILADO_25}' (3 passos); "
                        f"2.3 (--checkpoint) -> '{SIGMAS_DESTILADO_23}' (8 passos)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--saida", default="ltx", help="prefixo dos arquivos em ComfyUI/output")
    p.add_argument("--distorch", action="store_true",
                   help="distribui os blocos entre as placas em vez de descarregar inteiro. "
                        "Obrigatorio para um braco BF16 de 39 GiB, que sem isto da CUDA OOM")
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

    if not a.checkpoint and not a.transformer:
        raise SystemExit("modo 2.5 exige --transformer; modo 2.3 exige --checkpoint")
    if a.gguf and not a.checkpoint:
        raise SystemExit("--gguf so faz sentido com --checkpoint (VAEs e projecao vem dele)")
    if a.gguf and a.transformer:
        raise SystemExit("--gguf e --transformer sao dois modelos; escolha um")
    if not a.checkpoint and not a.video_vae:
        a.video_vae = "ltx-2.5-video-vae-bf16.safetensors"
    if a.sigmas is None:
        a.sigmas = SIGMAS_DESTILADO_23 if a.checkpoint else SIGMAS_DESTILADO_25
    if (a.frames - 1) % 8 != 0:
        raise SystemExit(f"--frames {a.frames} nao e 8n+1. Os validos ao redor sao "
                         f"{((a.frames - 1) // 8) * 8 + 1} e {((a.frames - 1) // 8 + 1) * 8 + 1}. "
                         "Este script recusa em vez de arredondar sozinho.")

    prompt = monta_prompt(a)
    modelo = a.gguf or a.transformer or a.checkpoint
    print(f"modelo      : {modelo}", flush=True)
    if a.checkpoint:
        print(f"checkpoint  : {a.checkpoint}", flush=True)
        print(f"audio vae   : {a.audio_checkpoint or a.checkpoint}   (FIXO entre bracos)", flush=True)
        print(f"projecao    : {a.proj_checkpoint or a.checkpoint}   (FIXO entre bracos)", flush=True)
    print(f"encoder     : {a.encoder}   (FIXO entre bracos)", flush=True)
    if a.lora:
        print(f"lora        : {a.lora} x{a.lora_strength} "
              f"({'bypass' if a.lora_bypass else 'fusao + requantizacao'})", flush=True)
    print(f"{a.frames} quadros a {a.fps} fps = {a.frames / a.fps:.2f} s, {a.size}px, "
          f"cfg {a.cfg}, semente {a.seed}, sigmas [{a.sigmas}]", flush=True)
    if a.distorch:
        print(f"distorch    : {a.alocacao or f'cuda:1,{a.doar_gb}gb;cpu,*'}", flush=True)

    t0 = time.time()
    pid = posta(a.servidor, prompt)
    print(f"prompt_id {pid}", flush=True)
    saida = espera(a.servidor, pid, a.limite, Path(__file__).resolve().parent.parent)
    segundos = time.time() - t0

    arq = arquivos_de_saida(saida)
    estado = saida.get("status", {})
    print(f"\nterminou em {segundos:.1f}s ({segundos / 60:.1f} min): "
          f"{len(arq['quadros'])} quadro(s), {len(arq['audio'])} audio(s), "
          f"{len(arq['video'])} video(s)", flush=True)
    for k in ("quadros", "audio", "video"):
        if arq[k]:
            print(f"  {k:<8}: {arq[k][0]}" + (f"  ..  {arq[k][-1]}" if len(arq[k]) > 1 else ""),
                  flush=True)
    if estado.get("status_str") and estado["status_str"] != "success":
        print(f"  STATUS {estado['status_str']}", flush=True)
    if not arq["audio"]:
        print("  ATENCAO: nenhum arquivo de audio na saida -- o ramo de audio nao foi gravado",
              flush=True)

    passos = len([s for s in a.sigmas.split(",") if s.strip()]) - 1
    reg = {"modelo": modelo, "transformer": a.transformer, "checkpoint": a.checkpoint,
           "audio_checkpoint": (a.audio_checkpoint or a.checkpoint) if a.checkpoint else None,
           "proj_checkpoint": (a.proj_checkpoint or a.checkpoint) if a.checkpoint else None,
           "gguf": a.gguf, "encoder": a.encoder, "video_vae": a.video_vae,
           "audio_vae": None if a.checkpoint else a.audio_vae,
           "lora": a.lora, "lora_strength": a.lora_strength if a.lora else None,
           "lora_bypass": bool(a.lora and a.lora_bypass),
           "frames": a.frames, "fps": a.fps, "size": a.size, "cfg": a.cfg,
           "sigmas": a.sigmas, "passos": passos, "seed": a.seed, "prompt": a.prompt,
           "negative": a.negative, "segundos": segundos, "s_por_quadro": segundos / a.frames,
           "s_por_passo": segundos / max(passos, 1),
           "quadros": arq["quadros"], "audio": arq["audio"], "video": arq["video"],
           "distorch": a.distorch,
           "alocacao": (a.alocacao or f"cuda:1,{a.doar_gb}gb;cpu,*") if a.distorch else None,
           "status": estado.get("status_str")}
    if a.json:
        Path(a.json).write_text(json.dumps(reg, indent=2), encoding="utf-8")
        print(f"registro em {a.json}", flush=True)

    print("\nNAO COBERTO: nada aqui julga a qualidade do video ou do audio, so que sairam e "
          "quanto demorou (`compara_av.py` mede). Nao confere que o encoder casa com o "
          "transformer. Nao le o log do servidor, entao um LoRA com chaves que nao casam passa "
          "aqui em silencio -- `probe_lora_requant.py` conta as chaves. O tempo inclui carga "
          "do modelo (`--cache-none` no servidor), nao e custo marginal por quadro.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
