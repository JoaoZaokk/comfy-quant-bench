"""Gera o notebook do Colab que roda o 10Eros v1.5 BF16 com o MESMO grafo do nosso `_2gpu` (I2V, DMD, 2 passadas),
para o dono comparar na tela com o render W4A8 local (mesmo prompt/seed/imagem).

Entrada: o grafo de API da rodada v2 (`prompt_2gpu_v2_api.json`). Os nós de constante (mxSlider, Seed, TwoWaySwitch,
CM_FloatToInt, ComfyMathExpression, PrimitiveFloat) viram valores literais e os nós só-locais (multi-GPU, Power Lora
Loader desligado, preview) saem, para o Colab precisar só de ComfyUI + KJNodes + VHS + 10S-Comfy-nodes.
Uso: python -s tools/colab_eros/monta_notebook.py <prompt_2gpu_v2_api.json> <saida.ipynb>
"""
import json
import sys

g = json.load(open(sys.argv[1]))
CK = "10Eros_v1.5_bf16.safetensors"


def troca_ref(no_velho, saida, valor):
    for v in g.values():
        for a, b in list(v["inputs"].items()):
            if isinstance(b, list) and b[0] == no_velho and (saida is None or b[1] == saida):
                v["inputs"][a] = valor


troca_ref("524", 0, "@SEED@")
troca_ref("540", 0, "@FPS_INT@")
troca_ref("542", 0, "@FPS@")
troca_ref("789", 0, ["556", 1])            # TwoWaySwitch em 2 = latente de audio da 1a passada
troca_ref("791", 0, "@LARGURA@")
troca_ref("792", 0, "@ALTURA@")
troca_ref("797", 0, 1.0)
troca_ref("798", 1, "@QUADROS@")
troca_ref("853", 0, ["906", 0])            # Power Lora Loader (tudo desligado) = passa direto
troca_ref("916", 0, 35)                    # compressao do LTXVPreprocess (mxSlider 35 -> CM_FloatToInt)
for k in ("524", "540", "542", "789", "791", "792", "796", "797", "798", "853", "876", "915", "916"):
    g.pop(k)
g["906"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "@LORA@", "strength_model": "@LORA_FORCA@",
                                                             "model": ["868", 0]}}
g["616"] = {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "@TE@", "ckpt_name": CK, "device": "default"}}
g["617"]["inputs"]["ckpt_name"] = CK
g["646"]["inputs"]["ckpt_name"] = CK
g["837"]["inputs"]["image"] = "@IMAGEM@"
g["536"]["inputs"]["text"] = "@PROMPT@"
neg_padrao = g["537"]["inputs"]["text"]
g["537"]["inputs"]["text"] = "@NEGATIVO@"
g["549"]["inputs"]["filename_prefix"] = "Eros/BF16_firstpass"
g["597"]["inputs"]["filename_prefix"] = "Eros/BF16"
restos = [(k, a, b) for k, v in g.items() for a, b in v["inputs"].items() if isinstance(b, list) and b[0] not in g]
assert not restos, restos
classes = sorted({v["class_type"] for v in g.values()})
TEMPLATE = json.dumps(g, ensure_ascii=False)


def code(src):
    return {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


INTRO = """
# 10Eros v1.5 **BF16** — I2V com DMD (mesmo workflow do `_2gpu` local)

Para comparar na tela com o render **W4A8** local: use **o mesmo prompt, seed, imagem, tamanho e quadros**.
O grafo é o do `10Eros_v1.5_W4A8_I2V_DMD_2gpu.json` (2 passadas, upscaler x2 1.1, decode em tiles), trocando só o
checkpoint pelo BF16 original e o text encoder pelo Gemma heretic BF16 (a fonte do nosso W4A8).

**Precisa de A100 com RAM alta** (Runtime → Change runtime type → A100, High-RAM). O checkpoint tem 43 GB e não cabe
inteiro em 40 GB de VRAM: o ComfyUI deixa uma parte na RAM da VM — é normal, só fica mais lento.
Downloads: ~70 GB, direto do Hugging Face para a VM. Rode as células 1 → 4; depois só a 4 para cada vídeo.

Nós usados: """ + ", ".join(f"`{c}`" for c in classes)

INSTALA = r'''
#@title 1. Instalar ComfyUI e nós (uma vez por VM)
import os, subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f"falhou: {c}")
os.chdir("/content")
if not os.path.isdir("/content/ComfyUI"):
    sh("git clone -q https://github.com/Comfy-Org/ComfyUI && cd ComfyUI && git checkout -q c1739380")
NOS = [("https://github.com/kijai/ComfyUI-KJNodes", "3f20054"),
       ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "4ee72c0"),
       ("https://github.com/TenStrip/10S-Comfy-nodes", "c231aca")]
for url, rev in NOS:
    d = "/content/ComfyUI/custom_nodes/" + url.rsplit("/", 1)[1]
    if not os.path.isdir(d):
        sh(f"git clone -q {url} {d} && cd {d} && git checkout -q {rev}")
# requirements do ComfyUI sem mexer no torch do Colab
def nome(l):
    for s in "=<>~![ ;":
        l = l.split(s)[0]
    return l.strip().lower()
req = [l for l in open("/content/ComfyUI/requirements.txt") if nome(l) not in ("torch", "torchvision", "torchaudio")]
open("/content/req_comfy.txt", "w").writelines(req)
sh(f"{sys.executable} -m pip install -q -r /content/req_comfy.txt")
for url, _ in NOS:
    r = "/content/ComfyUI/custom_nodes/" + url.rsplit("/", 1)[1] + "/requirements.txt"
    if os.path.isfile(r):
        sh(f"{sys.executable} -m pip install -q -r {r}")
sh("apt-get -qq install -y ffmpeg > /dev/null")
print("OK: ComfyUI e nós instalados")
'''

BAIXA = r'''
#@title 2. Baixar modelos do Hugging Face (~70 GB)
dmd = "LTX2.3_DMD_reshaped_r256.safetensors"  #@param ["LTX2.3_DMD_reshaped_r256.safetensors", "LTX2.3_DMD_hybrid_v2.safetensors"]
text_encoder = "gemma_3_12B_it_heretic.safetensors"  #@param ["gemma_3_12B_it_heretic.safetensors", "gemma_3_12B_it_heretic_fp8_e4m3fn.safetensors"]
import os, shutil, time
from huggingface_hub import hf_hub_download
M = "/content/ComfyUI/models"
ARQ = [("TenStrip/LTX2.3-10Eros", "10Eros_v1.5_bf16.safetensors", "checkpoints"),
       ("DreamFast/gemma-3-12b-it-heretic", "comfyui/" + text_encoder, "text_encoders"),
       ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "latent_upscale_models"),
       ("TenStrip/LTX2.3_DMD_Lora", dmd, "loras")]
# um DMD e um text encoder por vez: tira os outros das pastas para a celula 4 nao escolher o errado
for pasta, prefixo, escolhido in (("loras", "LTX2.3_DMD", dmd), ("text_encoders", "gemma_3_12B_it_heretic", text_encoder)):
    for f in os.listdir(f"{M}/{pasta}"):
        if f.startswith(prefixo) and f != escolhido:
            os.makedirs("/content/fora", exist_ok=True); shutil.move(f"{M}/{pasta}/{f}", f"/content/fora/{f}")
            print("guardado fora:", f)
    if os.path.isfile(f"/content/fora/{escolhido}"):
        shutil.move(f"/content/fora/{escolhido}", f"{M}/{pasta}/{escolhido}")
for repo, arq, pasta in ARQ:
    dest = f"{M}/{pasta}/{os.path.basename(arq)}"
    if os.path.isfile(dest):
        print("ja existe", dest); continue
    t0 = time.time()
    p = hf_hub_download(repo, arq, local_dir="/content/hf")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(p, dest)
    print(f"{dest}  {os.path.getsize(dest)/2**30:.1f} GiB em {time.time()-t0:.0f} s")
print("OK: modelos prontos")
'''

SOBE = r'''
#@title 3. Subir o ComfyUI em segundo plano
import subprocess, time, urllib.request, sys
LOG = open("/content/comfy.log", "a")
try:
    urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=2); print("ja estava no ar")
except Exception:
    # start_new_session: cancelar uma celula (Ctrl+M I / botao parar) NAO derruba o ComfyUI junto
    subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"], cwd="/content/ComfyUI",
                     stdout=LOG, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=2); break
        except Exception:
            time.sleep(5)
    else:
        print(open("/content/comfy.log").read()[-3000:]); raise SystemExit("ComfyUI nao subiu")
    print("ComfyUI no ar")
print(subprocess.run("grep -iE 'IMPORT FAILED|Cannot import' /content/comfy.log | tail -5", shell=True,
                     capture_output=True, text=True).stdout or "sem erro de import de nos")
'''

GERA = r'''
#@title 4. Gerar (use os MESMOS valores do render W4A8 local)
prompt = "A woman looks at the camera and says: \"Hello, this is a test of the audio.\" Natural lighting, subtle movement."  #@param {type:"string"}
negativo = @NEG_PADRAO@  #@param {type:"string"}
seed = 635141064074927  #@param {type:"integer"}
largura = 1024  #@param {type:"integer"}
altura = 1376  #@param {type:"integer"}
quadros = 360  #@param {type:"integer"}
fps = 24  #@param {type:"number"}
forca_dmd = 1.0  #@param {type:"number"}
enviar_imagem = True  #@param {type:"boolean"}
import json, os, time, glob, shutil, base64, urllib.request, urllib.error
from IPython.display import HTML, display
if enviar_imagem:
    from google.colab import files
    up = files.upload()
    nome_img = list(up)[0]
    shutil.move(nome_img, f"/content/ComfyUI/input/{nome_img}")
else:
    nome_img = max(os.listdir("/content/ComfyUI/input"), key=lambda f: os.path.getmtime(f"/content/ComfyUI/input/{f}"))
lora = [f for f in os.listdir("/content/ComfyUI/models/loras") if f.startswith("LTX2.3_DMD")][0]
te = [f for f in os.listdir("/content/ComfyUI/models/text_encoders") if f.startswith("gemma_3_12B_it_heretic")][0]
print("imagem:", nome_img, "| DMD:", lora, "| text encoder:", te)
TEMPLATE = json.loads(@TEMPLATE@)
VAL = {"@SEED@": int(seed), "@FPS@": float(fps), "@FPS_INT@": int(round(fps)), "@LARGURA@": int(largura),
       "@ALTURA@": int(altura), "@QUADROS@": int(quadros) + 1, "@LORA@": lora, "@LORA_FORCA@": float(forca_dmd),
       "@TE@": te, "@IMAGEM@": nome_img, "@PROMPT@": prompt, "@NEGATIVO@": negativo}
def preenche(x):
    if isinstance(x, dict): return {k: preenche(v) for k, v in x.items()}
    if isinstance(x, list): return [preenche(v) for v in x]
    return VAL.get(x, x) if isinstance(x, str) else x
grafo = preenche(TEMPLATE)
antes = set(glob.glob("/content/ComfyUI/output/Eros/BF16_*.mp4"))
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps({"prompt": grafo}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
except urllib.error.HTTPError as e:
    print(e.read().decode()[:3000]); raise
print("enviado", pid, "- acompanhe abaixo (a 1a vez le 43 GB do disco, demora)")
t0, visto = time.time(), len(open("/content/comfy.log", errors="replace").read())
def cancela():
    # parar a celula so para de acompanhar; isto cancela a geracao no ComfyUI sem matar o servidor
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/interrupt", data=b"", method="POST"))
    print("geracao cancelada no ComfyUI (servidor continua no ar)")
try:
  while True:
      time.sleep(15)
      log = open("/content/comfy.log", errors="replace").read()
      novo = log[visto:]; visto = len(log)
      for l in novo.splitlines():
          if any(s in l for s in ("loaded", "Requested", "Prompt executed", "Error", "Traceback", "out of memory")):
              print(f"[{time.time()-t0:5.0f}s] {l[:200]}")
      h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}").read())
      if pid in h:
          st = h[pid]["status"]; print("status:", st.get("status_str"), f"em {time.time()-t0:.0f} s")
          if st.get("status_str") != "success": print(json.dumps(st.get("messages", []))[-3000:])
          break
except KeyboardInterrupt:
    cancela(); raise
novos = sorted(set(glob.glob("/content/ComfyUI/output/Eros/BF16_*.mp4")) - antes)
final = [f for f in novos if "firstpass" not in f and not f.endswith("-audio.mp4")] or novos
for f in final:
    print(f)
    b64 = base64.b64encode(open(f, "rb").read()).decode()
    display(HTML(f'<video controls width="512" src="data:video/mp4;base64,{b64}"></video>'))
print("Para baixar: from google.colab import files; files.download(caminho) — ou o painel de arquivos à esquerda.")
'''

GERA = GERA.replace("@TEMPLATE@", repr(TEMPLATE)).replace("@NEG_PADRAO@", json.dumps(neg_padrao, ensure_ascii=False))
cells = [md(INTRO), code(INSTALA), code(BAIXA), code(SOBE), code(GERA)]
nb = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print("ok", sys.argv[2], "| nos:", classes)
