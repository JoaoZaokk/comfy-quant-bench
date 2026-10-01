"""Card do LTX 2.3: secao de como o braco BF16 foi obtido (seis mortes + mecanismo medido + rota GGUF)
e correcao do credito ao DisTorch2 (nao foi ele que fez o braco BF16 renderizar). A tabela de
medicao (TBD_MEASUREMENT_TABLE) entra depois, com os numeros."""
import io

p = 'bench/hf/ltx23-22b-w4a8/README.md'
s = io.open(p, encoding='utf-8').read()

def sub(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"anchor count {s.count(old)}: {old[:70]!r}")
    s = s.replace(old, new)

SECAO = """## How the BF16 reference was rendered, and what killed six attempts at it

The reference arm is the unquantized transformer: 39.13 GiB of BF16. On this machine — 63.6 GiB of
RAM, a 61 GiB system-managed pagefile, about 70 GiB of commit already taken by other processes —
six attempts to load it through ComfyUI's safetensors reader killed the server:

| # | loader | file read from | text encoder in the process | died in |
|---|---|---|---|---|
| 1–2 | `CheckpointLoaderSimple` + DisTorch2 `cpu,40gb`, the 43 GiB single file | W: (SMB share) | yes, live, 24.4 GB | `torch/storage.py __getitem__` inside `load_torch_file` |
| 3–4 | transformer extracted to its own file, `UNETLoaderDisTorch2MultiGPU` | W: | yes | same |
| 5 | same loader, saved conditioning, no encoder | C: (local NVMe) | no | `nn.Linear.__init__` — the model's `torch.empty` |
| 6 | dynamic VRAM on (lazy `Linear`, aimdo reader), no DisTorch | C: | no | `HostBuffer.read_file_slice failed`, `cudaErrorMemoryAllocation` |

Five of the six were `Windows fatal exception: access violation`. The network share was the first
suspect and it was cleared by measurement (2026-09-14, bare Python processes, system commit
counters read around each call): `safetensors.safe_open(framework="pt")` maps the file
**copy-on-write twice** — memmap2 for the header, `torch.UntypedStorage.from_file(shared=False)`
for the data — and Windows charges a copy-on-write view its whole size at mapping time, so opening
this 39 GiB file costs **+80.2 GiB of commit** before a tensor is read (+40.8 once the header view
is dropped), and building the model's parameters costs another +40.7. A read-only mmap costs 0.
When the charge forces the pagefile to grow, the new view sometimes comes back with the limit raised
but the charge not taken, and the first read through it is the access violation above — reproduced
in 20 seconds without ComfyUI, on the local drive as well as on the share; the same call survives
on other runs, which is why one such load in seven ever succeeded. A seventh attempt through the
same reader was not made: it would have needed about 117 GiB of commit against 53 free.

The way out was to not use that reader. `tools/safetensors_to_gguf_bf16.py` writes the same BF16
tensors into a GGUF container through a read-only memmap, with the type policy of ComfyUI-GGUF's
own converter (1-D, small, `scale_shift_table` and `learnable_registers` tensors in F32, exact
from BF16; everything else BF16) and the third-party Q6_K file as a template — 4444 of 4444 names
and shapes agree, and the F32 tensors are byte-identical to theirs in 32 of 32 sampled.
`UnetLoaderGGUF` reads it through `numpy.memmap` and assigns the tensors without `torch.empty`.
`tools/probe_gguf_bf16_equivalence.py` then checked, on 12 sampled Linear layers on the GPU, that
the bytes equal the safetensors', that ComfyUI-GGUF's BF16 dequantization returns the original
bf16 tensor bit for bit, and that `GGMLOps.Linear` and `comfy.ops.manual_cast.Linear` give
identical outputs for the same input, bias included: **12 of 12, maximum difference 0.0**. So the
reference arm is the BF16 model, executed through the GGUF loader; what that loader changes is
only where the weights live between steps (streamed from the page cache instead of resident),
which is a speed axis — its s/frame is reported and not compared.

"""

sub("""## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1_w4a8.safetensors`""",
SECAO + """## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1_w4a8.safetensors`""")

sub("""- **[city96 / ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — the loader for the
  third-party GGUF arm.""",
"""- **[city96 / ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — the loader for the
  third-party Q6_K arm, and the only way the 39 GiB BF16 reference could be loaded on this machine
  at all (read-only memmap, no `torch.empty`).""")

sub("""- **ComfyUI-MultiGPU (DisTorch2)** — the block splitting that let the 43 GiB BF16 reference arm
  render on a 24 GB card.
""",
"""- **ComfyUI-MultiGPU (DisTorch2)** — used in four of the six failed reference attempts; it was not
  what failed, and it is what rendered the 2.5 reference arm.
""")
io.open(p, 'w', encoding='utf-8').write(s)
print('card 2.3: mechanism section + credits patched')
