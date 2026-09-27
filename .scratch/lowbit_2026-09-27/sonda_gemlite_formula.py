"""Descobre a ordem de empacotamento e a formula do gemlite comparando contra o unpacked (BF16) do mesmo modelo.
Criterio: so vale a combinacao que reproduz o unpacked bit a bit; o controle (camada errada) tem de falhar."""
import json, struct, sys, torch
bits = int(sys.argv[1]); gem = sys.argv[2]; unp = sys.argv[3]
sd = torch.load(gem, map_location="cpu", weights_only=True, mmap=True)

def le_tensor(path, nome):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); e = h[nome]
        a, b = e["data_offsets"]; f.seek(8 + n + a); buf = f.read(b - a)
    return torch.frombuffer(bytearray(buf), dtype={"BF16": torch.bfloat16, "F32": torch.float32}[e["dtype"]]).view(e["shape"])

def camada(nome):
    Wq, s, z = sd[nome + ".W_q"], sd[nome + ".scales"], sd[nome + ".zeros"]
    meta = sd[nome + ".metadata"].tolist(); n, k = sd[nome + ".orig_shape"].tolist()
    return Wq, s, z, meta, n, k

def desempacota(Wq, bits, k, modo):
    r = 8 // bits; m = (1 << bits) - 1; kp, n = Wq.shape
    partes = [((Wq.to(torch.int32) >> (bits * i)) & m) for i in range(r)]  # cada uma (kp, n)
    if modo == "intercalado":   # linha kp*? : elemento i do byte = linha j*r+i
        return torch.stack(partes, 1).reshape(kp * r, n)
    if modo == "blocos":        # elemento i do byte = linha i*kp + j
        return torch.cat(partes, 0)
    if modo == "intercalado_msb":
        return torch.stack(partes[::-1], 1).reshape(kp * r, n)
    if modo == "blocos_msb":
        return torch.cat(partes[::-1], 0)

for nome in ["transformer_blocks.0.attn.to_q", "single_transformer_blocks.3.attn.to_out"]:
    Wq, s, z, meta, n, k = camada(nome)
    ref = le_tensor(unp, nome + ".weight").float()
    print(nome, "meta", meta, "shape", (n, k), "Wq", tuple(Wq.shape), "s", tuple(s.shape))
    g = k // s.shape[0]
    for modo in ["intercalado", "blocos", "intercalado_msb", "blocos_msb"]:
        q = desempacota(Wq, bits, k, modo).float()          # (k, n)
        sg = s.repeat_interleave(g, 0); zg = z.repeat_interleave(g, 0)
        for fn, w in {"(q-z)*s": (q - zg) * sg, "q*s+z": q * sg + zg}.items():
            w = w.t().to(torch.bfloat16).float()
            print(f"  {modo:16s} {fn:8s} identicos {(w == ref).float().mean().item():.6f}  relL2 {((w-ref).norm()/ref.norm()).item():.3e}")
# controle: camada certa do gemlite contra camada ERRADA do unpacked
Wq, s, z, meta, n, k = camada("transformer_blocks.0.attn.to_k")
ref = le_tensor(unp, "transformer_blocks.0.attn.to_q.weight").float()
