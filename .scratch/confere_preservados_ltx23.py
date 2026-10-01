"""Os tensores NAO quantizados (VAEs, vocoder, projecao) sao byte a byte iguais entre o BF16 e as
duas conversoes? E os arquivos soltos (projecao em text_encoders/, VAEs em vae/) sao os mesmos
bytes do checkpoint? Se sim, os bracos quantizados podem carregar VAEs/projecao de arquivos
pequenos em vez de mapear 43 GiB tres vezes por braco."""
import json, struct, hashlib, sys
def hdr(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q', f.read(8))[0]; h=json.loads(f.read(n)); h.pop('__metadata__',None); return h, 8+n
def sha_of(p, h, base, key):
    s,e = h[key]['data_offsets']
    m = hashlib.sha256()
    with open(p,'rb') as f:
        f.seek(base+s); left = e-s
        while left:
            b = f.read(min(left, 1<<24)); m.update(b); left -= len(b)
    return m.hexdigest()
BF = 'P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors'
files = {'w4a8': 'P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a8.safetensors',
         'w4a4': 'P:/ComfyBench/checkpoints/ltx-2.3-22b-distilled-1.1_w4a4.safetensors',
         'proj_solta': 'ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors',
         'audio_solto': 'ComfyUI/models/vae/LTX23_audio_vae_bf16.safetensors',
         'video_solto': 'ComfyUI/models/vae/LTX23_video_vae_bf16.safetensors'}
hb, bb = hdr(BF)
pres = [k for k in hb if k.split('.')[0] in ('vae','audio_vae','vocoder','text_embedding_projection')]
print(f"BF16 preserved tensors: {len(pres)}", flush=True)
ref = {k: sha_of(BF, hb, bb, k) for k in pres}
for name, p in files.items():
    h, base = hdr(p)
    same = diff = missing = 0
    for k in pres:
        kk = k
        if name == 'video_solto':      # o arquivo solto nao tem o prefixo `vae.`
            kk = k[len('vae.'):] if k.startswith('vae.') else None
            if kk is None: continue
        elif name in ('audio_solto',) and not (k.startswith('audio_vae.') or k.startswith('vocoder.')): continue
        elif name == 'proj_solta' and not k.startswith('text_embedding_projection.'): continue
        if kk not in h: missing += 1; continue
        if sha_of(p, h, base, kk) == ref[k]: same += 1
        else: diff += 1
    print(f"{name:<12} identical {same}  different {diff}  missing {missing}", flush=True)
