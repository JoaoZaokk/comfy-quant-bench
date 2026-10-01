"""Nunchaku com e sem dynamic VRAM num ComfyUI próprio (porta 8199, só a 3090).

    python teste_dynamic.py dyn|nodyn [casos]
Rodar sob Assert-GpuLock (roda.ps1).
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
BRACO = sys.argv[1]
OUT = D / (BRACO + os.environ.get("SUFIXO", ""))
PORTA = 8199
PRAZO = 1200
ZIMG_P = "A cozy bookshop interior at night, warm lamps, a cat sleeping on a stack of books, photo"
QWEN_P = "A neon shop sign that reads \"OPEN LATE\", rainy night, reflections on wet pavement, cinematic photo"


def grafo(modelo, clip, tipo, vae, prompt, seed, passos, shift):
    return {
        "1": modelo,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": tipo}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "seed": seed, "steps": passos, "cfg": 1.0,
              "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0], "negative": ["5", 0],
              "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}},
    }


CASOS = {
    "NZ": (lambda s: grafo({"class_type": "NunchakuZImageDiTLoader",
                            "inputs": {"model_name": "svdq-int4_r32-z-image-turbo.safetensors"}},
                           "qwen_3_4b.safetensors", "lumina2", "ae.safetensors", ZIMG_P, s, 8, 3.0), 5),
    "NQ": (lambda s: grafo({"class_type": "NunchakuQwenImageDiTLoader",
                            "inputs": {"model_name": "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors",
                                       "cpu_offload": "disable", "num_blocks_on_gpu": 1, "use_pin_memory": "disable"}},
                           "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", "qwen_image_vae.safetensors",
                           QWEN_P, s, 8, 3.1), 42),
    # controle: Z-Image comum (não Nunchaku), mesmo grafo
    "Z1": (lambda s: grafo({"class_type": "UNETLoader",
                            "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
                           "qwen_3_4b.safetensors", "lumina2", "ae.safetensors", ZIMG_P, s, 8, 3.0), 5),
}


def main():
    casos = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CASOS)
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", COMFYUI_MGPU_DISABLED="1",
               PYTHONNOUSERSITE="1")
    args = [str(RAIZ / "python_embeded/python.exe"), "-s", str(RAIZ / "ComfyUI/main.py"), "--windows-standalone-build",
            "--use-sage-attention", "--enable-triton-backend", "--listen", "127.0.0.1", "--port", str(PORTA)]
    if BRACO == "nodyn":
        args.append("--disable-dynamic-vram")
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
                g["9"]["inputs"]["filename_prefix"] = f"dynamic_2026-10-01/{OUT.name}/{caso}_{rodada}_s{s}"
                reg = cc.roda_um(comfy, g, PRAZO, rotulo=caso,
                                 extra={**ctl, "caso": caso, "rodada": rodada, "seed": s, "braco": BRACO})
                print(json.dumps(reg, ensure_ascii=False, default=str)[:600], flush=True)
                cc.grava_jsonl(OUT / "resultados.jsonl", reg)
                regs.append(reg)
        return cc.codigo_de_saida(regs)
    finally:
        r = subprocess.run(["taskkill", "/PID", str(srv.pid), "/T", "/F"], capture_output=True, text=True)
        print(f"taskkill rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}", flush=True)
        srv.kill()
        srv.wait(timeout=60)
        # o servidor pode ter sido relançado com outro PID (filho do PID do Popen); mata pelo argumento da porta
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                        f"? {{ $_.CommandLine -match '--port {PORTA}' }} | % {{ Stop-Process -Id $_.ProcessId -Force }}"],
                       capture_output=True)
        for _ in range(30):
            try:
                comfy.system_stats(timeout=2)
            except Exception:
                print("servidor derrubado", flush=True)
                break
            time.sleep(1)
        else:
            print("AVISO: porta ainda responde 30 s depois do taskkill", flush=True)
            sys.exit(9)


if __name__ == "__main__":
    sys.exit(main())
