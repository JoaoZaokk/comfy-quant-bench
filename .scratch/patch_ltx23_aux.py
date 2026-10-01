"""Divide o loader auxiliar do ltx_video.py em audio/projecao (arquivos pequenos) e reescreve a
fila do 2.3 para usar a copia local do BF16 em W: e os arquivos pequenos."""
import ast
import io
import shutil
import os

# ---- arquivos pequenos em checkpoints/ (copias; os originais ficam onde estao)
for src in ("ComfyUI/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
            "ComfyUI/models/vae/LTX23_audio_vae_bf16.safetensors"):
    dst = os.path.join("P:/ComfyBench/checkpoints", os.path.basename(src))
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copyfile(src, dst)
    print("checkpoints/:", os.path.basename(dst), os.path.getsize(dst), "B")

# ---- ltx_video.py
p = 'tools/ltx_video.py'
s = io.open(p, encoding='utf-8').read()
old = """        # `--aux-checkpoint` e de onde saem os VAEs e a projecao de texto (default: o proprio
        # --checkpoint). Um braco quantizado passa `--checkpoint <w4a8> --aux-checkpoint <bf16>`:
        # o modelo vem do arquivo quantizado pelo caminho de checkpoint (que le
        # `_quantization_metadata`, como faz com o `dev-fp8` de fabrica), e VAEs/projecao vem
        # SEMPRE do BF16 -- sao byte a byte iguais no arquivo convertido, mas assim o eixo fica
        # puro por construcao e nao por confianca no conversor.
        aux = a.aux_checkpoint or a.checkpoint
        if a.distorch:
            g["1c"] = {"class_type": "CheckpointLoaderSimpleDisTorch2MultiGPU",
                       "inputs": {"ckpt_name": a.checkpoint, "compute_device": "cuda:0",
                                  "expert_mode_allocations": aloc, "eject_models": True}}
        else:
            g["1c"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.checkpoint}}
        if aux != a.checkpoint:
            g["1a"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": aux}}
"""
new = """        # Cada loader auxiliar le o arquivo INTEIRO que recebe (`load_torch_file`), entao apontar
        # `LTXVAudioVAELoader` e `LTXAVTextEncoderLoader` para o checkpoint de 43 GiB mapeia 43 GiB
        # duas vezes a mais por braco -- e foi um mmap de 39 GiB lido do SMB que derrubou o servidor
        # no 2.5. `--audio-checkpoint` e `--proj-checkpoint` apontam para arquivos pequenos em
        # `checkpoints/` com SO o VAE de audio + vocoder (1329 tensores) e SO a projecao (4).
        # Conferido byte a byte em 2026-09-13: os 1503 tensores preservados sao identicos entre o
        # BF16, o W4A8 e o W4A4, e os arquivos soltos sao identicos ao checkpoint. Default: o
        # proprio --checkpoint. O VAE de VIDEO sai do mesmo loader do modelo (saida 2).
        audio_ck = a.audio_checkpoint or a.checkpoint
        proj_ck = a.proj_checkpoint or a.checkpoint
        if a.distorch:
            g["1c"] = {"class_type": "CheckpointLoaderSimpleDisTorch2MultiGPU",
                       "inputs": {"ckpt_name": a.checkpoint, "compute_device": "cuda:0",
                                  "expert_mode_allocations": aloc, "eject_models": True}}
        else:
            g["1c"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.checkpoint}}
"""
assert old in s
s = s.replace(old, new)
old2 = """        g["2"] = {"class_type": "LTXAVTextEncoderLoader",
                  "inputs": {"text_encoder": a.encoder, "ckpt_name": aux, "device": "default"}}
        if a.video_vae:
            g["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}}
            vae_video = ["5", 0]
        else:
            vae_video = ["1a", 2] if aux != a.checkpoint else ["1c", 2]
        g["6"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": aux}}
"""
new2 = """        g["2"] = {"class_type": "LTXAVTextEncoderLoader",
                  "inputs": {"text_encoder": a.encoder, "ckpt_name": proj_ck, "device": "default"}}
        if a.video_vae:
            g["5"] = {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}}
            vae_video = ["5", 0]
        else:
            vae_video = ["1c", 2]
        g["6"] = {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": audio_ck}}
"""
assert old2 in s
s = s.replace(old2, new2)
old3 = """    p.add_argument("--aux-checkpoint", default=None,
                   help="com --checkpoint: de onde vem VAEs e projecao de texto (default: o proprio "
                        "--checkpoint). Braco quantizado: --checkpoint <w4a8> --aux-checkpoint <bf16>")
"""
new3 = """    p.add_argument("--audio-checkpoint", default=None,
                   help="com --checkpoint: arquivo em checkpoints/ de onde LTXVAudioVAELoader le o VAE "
                        "de audio + vocoder (default: o proprio --checkpoint; um arquivo pequeno so com "
                        "audio_vae.*/vocoder.* evita mapear o checkpoint inteiro de novo)")
    p.add_argument("--proj-checkpoint", default=None,
                   help="com --checkpoint: arquivo em checkpoints/ de onde LTXAVTextEncoderLoader le a "
                        "projecao de texto (default: o proprio --checkpoint)")
"""
assert old3 in s
s = s.replace(old3, new3)
old4 = """           "aux_checkpoint": (a.aux_checkpoint or a.checkpoint) if a.checkpoint else None,"""
new4 = """           "audio_checkpoint": (a.audio_checkpoint or a.checkpoint) if a.checkpoint else None,
           "proj_checkpoint": (a.proj_checkpoint or a.checkpoint) if a.checkpoint else None,"""
assert old4 in s
s = s.replace(old4, new4)
old5 = """        print(f"aux         : {a.aux_checkpoint or a.checkpoint}   (VAEs e projecao de texto, FIXO entre bracos)", flush=True)"""
new5 = """        print(f"audio vae   : {a.audio_checkpoint or a.checkpoint}   (FIXO entre bracos)", flush=True)
        print(f"projecao    : {a.proj_checkpoint or a.checkpoint}   (FIXO entre bracos)", flush=True)"""
assert old5 in s
s = s.replace(old5, new5)
ast.parse(s)
io.open(p, 'w', encoding='utf-8').write(s)
print('ltx_video.py: audio/proj checkpoints, syntax ok')

# ---- fila_ltx23.sh
p = '.scratch/fila_ltx23.sh'
s = io.open(p, encoding='utf-8').read()
gate = "until grep -q 'GEMMA3_TE_DOWNLOAD_OK' .scratch/baixa_gemma3_te.log 2>/dev/null; do sleep 30; done\n"
assert gate in s
s = s.replace(gate, gate +
    "# a copia do BF16 para W: (disco local) tem de estar inteira: 46149345334 B\n"
    'until [ "$(stat -c %s W:/ltx-2.3/ltx-2.3-22b-distilled-1.1_W.safetensors 2>/dev/null)" = "46149345334" ]; do sleep 30; done\n')
s = s.replace("CK=ltx-2.3-22b-distilled-1.1.safetensors\n",
              "CK=ltx-2.3-22b-distilled-1.1_W.safetensors\n"
              'AUX="--audio-checkpoint LTX23_audio_vae_bf16.safetensors --proj-checkpoint ltx-2.3_text_projection_bf16.safetensors"\n')
s = s.replace('run() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py --checkpoint $CK --encoder $ENC --frames 249',
              'run() { echo "--- $1 $(date)"; $PY -s tools/ltx_video.py $AUX --encoder $ENC --frames 249')
s = s.replace('run ltx23av_w4a8 --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors --aux-checkpoint $CK',
              'run ltx23av_w4a8 --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors')
s = s.replace('run ltx23av_w4a4 --checkpoint ltx-2.3-22b-distilled-1.1_w4a4.safetensors --aux-checkpoint $CK',
              'run ltx23av_w4a4 --checkpoint ltx-2.3-22b-distilled-1.1_w4a4.safetensors')
s = s.replace('run ltx23av_q6k  --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf',
              'run ltx23av_q6k  --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors --gguf ltx-2.3-22b-distilled-1.1-Q6_K.gguf')
s = s.replace('run ltx23av_bf16 --distorch --alocacao "cpu,40gb" --limite 10800',
              'run ltx23av_bf16 --checkpoint $CK --distorch --alocacao "cpu,40gb" --limite 10800')
s = s.replace('$PY -s tools/ltx_video.py --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors --aux-checkpoint $CK --encoder $ENC --frames 49',
              '$PY -s tools/ltx_video.py --checkpoint ltx-2.3-22b-distilled-1.1_w4a8.safetensors $AUX --encoder $ENC --frames 49')
s = s.replace("grep -o 'ltx-2.3-22b-distilled-1.1[a-z0-9_.-]*' | sort -u",
              "grep -o -E 'ltx-2.3[A-Za-z0-9_.-]*|LTX23_audio[A-Za-z0-9_.-]*' | sort -u")
assert 'aux-checkpoint' not in s, 'aux-checkpoint left in the queue'
io.open(p, 'w', encoding='utf-8').write(s)
print('fila_ltx23.sh patched')
for line in s.splitlines():
    if line.startswith(('until', 'CK=', 'AUX=', 'run ', 'run()', 'runl(', 'runl ')):
        print('  ', line[:190])
