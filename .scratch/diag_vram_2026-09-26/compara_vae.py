"""O VAE de video avulso (LTX23_video_vae_bf16) e byte a byte o vae.* do checkpoint 10Eros? Le por offset, sem mmap."""
import hashlib, json, struct, sys
def hdr(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    return f, 8 + n, h
ck, av = sys.argv[1], sys.argv[2]
fc, dc, hc = hdr(ck); fa, da, ha = hdr(av)
vc = {k[4:]: v for k, v in hc.items() if k.startswith("vae.")}
print("checkpoint vae.*:", len(vc), " avulso:", len(ha))
print("so no ckpt:", sorted(set(vc) - set(ha))[:5], " so no avulso:", sorted(set(ha) - set(vc))[:5])
dif = 0
for k in sorted(set(vc) & set(ha)):
    a, b = vc[k], ha[k]
    if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
        dif += 1; print("forma/dtype", k, a["dtype"], a["shape"], b["dtype"], b["shape"]); continue
    fc.seek(dc + a["data_offsets"][0]); x = fc.read(a["data_offsets"][1] - a["data_offsets"][0])
    fa.seek(da + b["data_offsets"][0]); y = fa.read(b["data_offsets"][1] - b["data_offsets"][0])
    if x != y:
        dif += 1; print("bytes diferem", k)
print("tensores diferentes:", dif)
