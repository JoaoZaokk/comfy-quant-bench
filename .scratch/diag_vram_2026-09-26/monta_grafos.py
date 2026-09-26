"""base = grafo da rodada audioint8 de 25/09 (VAE do checkpoint e VAE de audio na cuda:0).
vae1 = igual, mas VAE de video (avulso, byte a byte igual ao do checkpoint) e VAE de audio na cuda:1."""
import copy, json, sys
src, dst = sys.argv[1], sys.argv[2]
g = json.load(open(src))
def prefixo(gr, nome):
    for n in gr.values():
        if n["class_type"] == "VHS_VideoCombine":
            fp = n["inputs"]["filename_prefix"]
            n["inputs"]["filename_prefix"] = fp.replace("I2V_DMD_audioint8", "diag_" + nome)
base = copy.deepcopy(g); prefixo(base, "base")
v = copy.deepcopy(g); prefixo(v, "vae1")
v["950"] = {"class_type": "VAELoaderMultiGPU", "inputs": {"vae_name": "LTX23_video_vae_bf16.safetensors", "device": "cuda:1"}}
troca = 0
for n in v.values():
    for k, x in n["inputs"].items():
        if x == ["646", 2]:
            n["inputs"][k] = ["950", 0]; troca += 1
v["617"] = {"class_type": "LTXV2AudioVAELoaderMultiGPU", "inputs": {"ckpt_name": "LTX23_audio_vae_bf16.safetensors", "device": "cuda:1"}}
print("refs do VAE trocadas:", troca)
json.dump(base, open(f"{dst}/prompt_base_api.json", "w"), indent=1)
json.dump(v, open(f"{dst}/prompt_vae1_api.json", "w"), indent=1)
