"""Chaves, shapes e dtypes do pack gemlite (state_dict.pt) e do MLX, sem desquantizar nada."""
import collections, json, sys, torch
p = sys.argv[1]
sd = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
print(type(sd), len(sd))
tipos = collections.Counter()
exemplo = {}
for k, v in sd.items():
    suf = k.rsplit(".", 1)[-1]
    desc = (type(v).__name__, str(getattr(v, "dtype", "")), tuple(getattr(v, "shape", ())) if hasattr(v, "shape") else repr(v)[:60])
    tipos[suf] += 1
    base = k.rsplit(".", 1)[0]
    if base.startswith("transformer_blocks.0.attn.to_q") or base.startswith("single_transformer_blocks.0.attn.to_out") or base == "proj_out":
        exemplo[k] = desc
print(dict(tipos))
for k, d in exemplo.items():
    print(k, d)
