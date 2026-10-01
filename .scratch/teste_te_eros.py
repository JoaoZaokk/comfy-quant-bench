"""Renders do Eros (ultimo prompt, seed fixa) com tres pares de text encoder, via API do ComfyUI 8190.
So' troca os dois arquivos do no 616 e os prefixos de saida. Nao abre imagem nem audio: compara depois
numericamente (teste_te_eros_mede.py). Uso: python -s .scratch/teste_te_eros.py <prompt_api.json>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8190"
BRACOS = {
    "base": ("gemma_3_12B_it_heretic_w4a8.safetensors", "ltx-2.3_text_projection_bf16.safetensors"),
    "fp8": ("gemma_3_12B_it_heretic_w4a8_embfp8.safetensors", "ltx-2.3_text_projection_fp8.safetensors"),
    "int8": ("gemma_3_12B_it_heretic_w4a8_embint8.safetensors", "ltx-2.3_text_projection_int8.safetensors"),
}


def req(path, data=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


def main():
    grafo = json.load(open(sys.argv[1]))
    while True:
        try:
            req("/queue")
            break
        except Exception:
            time.sleep(5)
    for braco, (te, proj) in BRACOS.items():
        g = json.loads(json.dumps(grafo))
        g["616"]["inputs"]["text_encoder"] = te
        g["616"]["inputs"]["ckpt_name"] = proj
        g["597"]["inputs"]["filename_prefix"] = f"Eros/teste_te_{braco}"
        g["549"]["inputs"]["filename_prefix"] = f"Eros/teste_te_{braco}_firstpass"
        t0 = time.time()
        pid = req("/prompt", {"prompt": g})["prompt_id"]
        print(f"{braco}: enviado {pid}", flush=True)
        while True:
            h = req(f"/history/{pid}")
            if pid in h:
                st = h[pid]["status"]
                print(f"{braco}: {st.get('status_str')} em {time.time() - t0:.0f} s", flush=True)
                if st.get("status_str") != "success":
                    print(json.dumps(st.get("messages", []))[-2000:], flush=True)
                break
            time.sleep(10)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
