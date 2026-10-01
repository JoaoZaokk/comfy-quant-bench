"""ltx_video.py: em modo --checkpoint, so carrega o checkpoint (no 1c) se algo precisar dele --
modelo (sem --transformer/--gguf) ou VAE de video (sem --video-vae). Com --transformer +
--video-vae + --audio-checkpoint + --proj-checkpoint, o checkpoint de 43 GiB nao e tocado."""
import ast
import io

p = 'tools/ltx_video.py'
s = io.open(p, encoding='utf-8').read()

old = """        audio_ck = a.audio_checkpoint or a.checkpoint
        proj_ck = a.proj_checkpoint or a.checkpoint
        if a.distorch:
            g["1c"] = {"class_type": "CheckpointLoaderSimpleDisTorch2MultiGPU",
                       "inputs": {"ckpt_name": a.checkpoint, "compute_device": "cuda:0",
                                  "expert_mode_allocations": aloc, "eject_models": True}}
        else:
            g["1c"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.checkpoint}}
"""
new = """        audio_ck = a.audio_checkpoint or a.checkpoint
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
"""
assert old in s
s = s.replace(old, new)
ast.parse(s)
io.open(p, 'w', encoding='utf-8').write(s)
print('ltx_video.py: 1c only when needed; syntax ok')
