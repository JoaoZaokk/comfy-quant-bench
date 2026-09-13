"""Codifica e decodifica a imagem no VAE, SEM amostrar. Isola o round-trip do sampler.

POR QUE: a deriva de textura sobreviveu a denoise 0.4, entao nao e a amostragem. O unico outro
caminho pelo qual a imagem passa e o VAE: VAEEncode -> (sampler) -> VAEDecode. Se codificar e
decodificar sozinho ja destroi a madeira, o mecanismo esta achado e nao tem nada a ver com
quantizacao -- o VAE nao e quantizado nesta bancada, por regra do dono.
"""
import json, urllib.request, time, sys

G = {
 "1": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
 "2": {"class_type": "LoadImage", "inputs": {"image": "edit_maca.png"}},
 "3": {"class_type": "ImageScaleToTotalPixels",
       "inputs": {"image": ["2",0], "upscale_method": "lanczos",
                  "megapixels": 1.0, "resolution_steps": 1}},
 "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3",0], "vae": ["1",0]}},
 "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4",0], "vae": ["1",0]}},
 "6": {"class_type": "SaveImage", "inputs": {"images": ["5",0], "filename_prefix": "vae_roundtrip"}},
}
req = urllib.request.Request("http://127.0.0.1:8190/prompt",
                             data=json.dumps({"prompt": G}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
except Exception as e:
    print("RECUSADO:", e); sys.exit(1)
t0 = time.time()
while time.time()-t0 < 600:
    h = json.load(urllib.request.urlopen(f"http://127.0.0.1:8190/history/{pid}", timeout=30))
    if pid in h:
        outs=[i["filename"] for n in h[pid].get("outputs",{}).values() for i in n.get("images",[])]
        print(f"pronto em {time.time()-t0:.1f}s:", outs); sys.exit(0)
    time.sleep(3)
print("expirou"); sys.exit(1)
