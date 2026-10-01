"""A condicao de texto do ComfyUI (a que o render e o epsilon usam) contra as dos shards do professor:
local (diffusers 0.38 / transformers 4.57, 3090) e VM (diffusers 0.40 / transformers 5.16, A100).
Mesmo prompt. Pergunta: o professor da VM treina o aluno numa condicao diferente da do uso?"""
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "tools"))
import quality_ladder  # noqa: E402,F401  -- so' o preambulo que prepara o ComfyUI como biblioteca
import torch  # noqa: E402
import comfy.sd as comfy_sd  # noqa: E402
import folder_paths  # noqa: E402

v = torch.load("F:/qat_klein/shard_vm/p0000_s1.pt", weights_only=False)
lo = torch.load("F:/qat_klein/run_4bit/professor_holdout/p0000_s1.pt", weights_only=False)
clip = comfy_sd.load_clip(ckpt_paths=[folder_paths.get_full_path_or_raise("text_encoders", "qwen_3_4b.safetensors")],
                          embedding_directory=folder_paths.get_folder_paths("embeddings"),
                          clip_type=comfy_sd.CLIPType.FLUX2, disable_dynamic=True)
c = clip.encode_from_tokens_scheduled(clip.tokenize(v["prompt"]))[0][0].float().cpu()
ev, el = v["comum"]["encoder_hidden_states"].float(), lo["comum"]["encoder_hidden_states"].float()
print("comfy", tuple(c.shape), "vm", tuple(ev.shape), "local", tuple(el.shape))


def rel(a, b):
    return float((a - b).norm() / b.norm())


def normas(t):
    n = t[0].norm(dim=-1)
    return n


nv, nl, nc = normas(ev), normas(el), normas(c)
print("normas por token (primeiros 12) vm   ", [round(float(x), 1) for x in nv[:12]])
print("normas por token (primeiros 12) local", [round(float(x), 1) for x in nl[:12]])
print("normas por token (primeiros 12) comfy", [round(float(x), 1) for x in nc[:12]])
L = min(c.shape[1], ev.shape[1])
print(f"sobreposicao {L} tokens")
print(f"rel vm vs local  (todos 512)  {rel(ev, el):.4e}")
print(f"rel comfy vs local (primeiros {L}) {rel(c[:, :L], el[:, :L]):.4e}")
print(f"rel comfy vs vm    (primeiros {L}) {rel(c[:, :L], ev[:, :L]):.4e}")
for n in (8, 16, 32):
    print(f"  primeiros {n}: comfy-local {rel(c[:, :n], el[:, :n]):.4e}  comfy-vm {rel(c[:, :n], ev[:, :n]):.4e}"
          f"  vm-local {rel(ev[:, :n], el[:, :n]):.4e}")
