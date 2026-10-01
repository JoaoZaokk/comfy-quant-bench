"""Roda os grafos do critério com a flag ON ou OFF num ComfyUI próprio (porta 8199, só a 3090 visível).

    python valida_triton.py on|off [casos,separados]
Resultado: <flag>/resultados.jsonl, <flag>/contador.json, <flag>/comfy.log. Rodar sob Assert-GpuLock.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "tools"))
import comfy_client as cc  # noqa: E402

D = pathlib.Path(__file__).resolve().parent
FLAG = sys.argv[1]
OUT = D / FLAG
PORTA = 8199
PRAZO = 1500
BAT = RAIZ / ".scratch/qwen21_2026-09-26/bateria"
KLEIN = "klein4b_braco2_bonsai_ternario_bfl.safetensors"
KLEIN_MLX = "F:/bonsai-re/bonsai-image-ternary-4B-mlx-2bit/transformer-packed-mflux/diffusion_pytorch_model.safetensors"
KLEIN_P0 = "a red apple on a weathered wooden table, soft window light"
ZIMG_P = "A cozy bookshop interior at night, warm lamps, a cat sleeping on a stack of books, photo"


def qwen(braco, seed):
    g = json.loads((BAT / f"{braco}_p0_s42.json").read_text(encoding="utf-8"))
    g = g.get("prompt", g)
    g["6"]["inputs"]["seed"] = seed
    return g


def texto_para_imagem(modelo, clip, tipo, vae, latente, prompt, seed, passos, shift=None):
    g = {
        "1": modelo,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": tipo}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": latente, "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": passos, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0],
              "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
    }
    if shift is not None:
        g["10"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift}}
        g["7"]["inputs"]["model"] = ["10", 0]
    return g


def unet(nome):
    return {"class_type": "UNETLoader", "inputs": {"unet_name": nome, "weight_dtype": "default"}}


def klein(mlx, seed):
    modelo = ({"class_type": "LowBitDiffusionLoader", "inputs": {"unet_name": KLEIN, "device": "cuda:0",
               "offload_device": "auto", "compute_dtype": "auto", "path": KLEIN_MLX}} if mlx else unet(KLEIN))
    return texto_para_imagem(modelo, "qwen_3_4b.safetensors", "flux2", "flux2_klein_vae_diffusers.safetensors",
                             "EmptyFlux2LatentImage", KLEIN_P0, seed, 4)


def zimage(nome, seed):
    return texto_para_imagem(unet(nome), "qwen_3_4b.safetensors", "lumina2", "ae.safetensors",
                             "EmptySD3LatentImage", ZIMG_P, seed, 8, shift=3.0)


# ordem: modelos locais (F:) primeiro, depois os do NAS. Q2 (BF16 14 GB do NAS) fica de fora por padrão: é o caso
# que abortou o dynamic VRAM em 27/09 e prendeu a placa (memória dynamic-vram-nas-e-quant-fp32).
CASOS = {
    "K1": (lambda s: klein(False, s), 11), "K2": (lambda s: klein(True, s), 11),
    "Z1": (lambda s: zimage("z_image_turbo_bf16.safetensors", s), 5),
    "Z2": (lambda s: zimage("zimage_turbo_w4a4.safetensors", s), 5),
    "Q1": (lambda s: qwen("w4a16_q4_1_nativo", s), 42), "Q2": (lambda s: qwen("bf16", s), 42),
    "Q3": (lambda s: qwen("nosso_w4a8", s), 42), "Q4": (lambda s: qwen("nosso_w4a4", s), 42),
    "Q5": (lambda s: qwen("nosso_int8", s), 42),
}
PADRAO = [c for c in CASOS if c != "Q2"]


def main():
    casos = sys.argv[2].split(",") if len(sys.argv) > 2 else PADRAO
    OUT.mkdir(parents=True, exist_ok=True)
    # COMFYUI_MGPU_DISABLED: sem isso o multigpu-orchestrator manda o prompt a um worker que não herda as flags
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CONTADOR_SAIDA=str(OUT / "contador.json"), PYTHONUNBUFFERED="1",
               COMFYUI_MGPU_DISABLED="1", PYTHONNOUSERSITE="1")
    args = [str(RAIZ / "python_embeded/python.exe"), "-s", str(D / "comfy_contador.py"), "--windows-standalone-build",
            "--use-sage-attention", "--disable-pinned-memory", "--listen", "127.0.0.1", "--port", str(PORTA)]
    if FLAG == "on":
        args.append("--enable-triton-backend")
    comfy = cc.Comfy(f"127.0.0.1:{PORTA}")
    try:
        comfy.system_stats(timeout=3)
        raise SystemExit(f"porta {PORTA} já responde; não é meu servidor")
    except SystemExit:
        raise
    except Exception:
        pass
    with open(OUT / "comfy.log", "a", encoding="utf-8") as log:
        srv = subprocess.Popen(args, cwd=RAIZ, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        comfy.wait_up(600)
        ctl = cc.controles(comfy)
        regs = []
        for caso in casos:
            fazer, seed = CASOS[caso]
            for rodada, s in (("frio", seed), ("quente", seed + 1)):
                g = fazer(s)
                for no in g.values():
                    if no["class_type"] == "SaveImage":
                        no["inputs"]["filename_prefix"] = f"triton_2026-09-29/{FLAG}/{caso}_{rodada}_s{s}"
                reg = cc.roda_um(comfy, g, PRAZO, rotulo=caso,
                                 extra={**ctl, "caso": caso, "rodada": rodada, "seed": s, "flag": FLAG})
                print(json.dumps(reg, ensure_ascii=False, default=str)[:400], flush=True)
                cc.grava_jsonl(OUT / "resultados.jsonl", reg)
                regs.append(reg)
        return cc.codigo_de_saida(regs)
    finally:
        subprocess.run(["taskkill", "/PID", str(srv.pid), "/T", "/F"], capture_output=True)
        srv.wait(timeout=60)
        time.sleep(3)
        try:
            comfy.system_stats(timeout=3)
            print("AVISO: porta ainda responde depois do taskkill", flush=True)
            sys.exit(9)
        except SystemExit:
            raise
        except Exception:
            print("servidor derrubado", flush=True)


if __name__ == "__main__":
    sys.exit(main())
