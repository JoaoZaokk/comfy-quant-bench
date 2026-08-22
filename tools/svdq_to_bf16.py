"""Recover BF16 weights from an SVDQuant (Nunchaku) checkpoint.

Why this exists: some models are published *only* as SVDQuant. A community fine-tune like
`Beyond_Reality_Zimage_v2` or a third-party build like `Qwen-Image-2512` has no BF16 upstream to
fall back on, so it can run under Nunchaku and nowhere else -- no GGUF, no OpenVINO, no LoRA
merge, no Arc. This turns one back into an ordinary BF16 safetensors file.

**It keeps the quantised file's FUSED layer layout.** Measured on the full Beyond_Reality
Z-Image: the recovery writes 419 tensors with 68 fused keys (`attention.to_qkv`,
`feed_forward.net.0.proj`) where the published BF16 has 521 tensors and none. The bytes are
equivalent; the names are not. A loader that expects separate `to_q`/`to_k`/`to_v` and `w1`/`w3`
will not find them, so "any loader can read it" is only true for loaders that already accept the
fused form. Splitting them back is possible -- the shapes say where the cuts go -- but is not
implemented and has never been needed here.

**What you get back is not the original.** Dequantising does not restore information: the result
carries INT4 quality at BF16 size.

But it is NOT pointless for a model whose BF16 you already have, which is what an earlier version
of this paragraph claimed. The recovered file has the quantised **weights** and full-precision
**activations**, so running original / recovered / INT4 on one seed isolates the two halves of
W4A4:

    original    BF16 weights   BF16 activations
    recovered   INT4 weights   BF16 activations   <- W4 only
    INT4        INT4 weights   INT4 activations   <- W4 + A4

Observed on Beyond_Reality Z-Image v2, one seed, eyeballed: a thin pencil survives the recovered
run intact and is deformed in the INT4 run. Same weights in both -- the identity probe is exact --
so the difference is activation quantisation alone. Weight error is a fixed offset that bends the
trajectory smoothly; activation error is fresh high-frequency noise at every block of every step,
and thin few-pixel structure is where it shows first. Consistent with `feed_forward.net.2`, whose
input is post-activation, scoring worst in every diagnostic here.

Caveat, so nobody quotes this as settled: in latent space the two sit at the SAME distance from
the original (relL2 0.4019 recovered, 0.4046 INT4). The *kind* of error differs, not the
magnitude, and the perceptual half of that is one image on one seed.

## Why probing instead of unpacking

`qweight` is not "two int4 per byte in order". It is a tensor-core lane interleave whose layout
depends on `warp_n`, `insn_k` and `s_pack_size`, and the package ships no inverse -- every
converter in `nunchaku/` goes *into* the format, none comes out. `packer.py` documents the
scale ordering as, in part:

    #  0   32  64  96   <-- load by lane 0
    #  8   40  72  104  <-- load by lane 1

Reimplementing that inverse means tracking a layout that belongs to the kernel and can change
with it. So this does not try. For any linear map, feeding the identity recovers the matrix:

    W.T == layer.forward(I)

which delegates the entire unpacking question to the kernel that owns it. What comes back is the
weight the kernel actually holds, expressed through the kernel's own activation path.

**How exact, honestly.** This block used to say "exact, not approximate ... with no error at
all". The *weight* side supports that: each row of `I` holds one nonzero, and a symmetric
per-group scale represents a lone nonzero and its neighbouring zeros without error. The
*activation* side does not, and the evidence offered was weaker than the claim. What was actually
measured on the real kernel is homogeneity -- `layer(8I)/8` matches `layer(I)` to exactly 0 on
every layer sampled -- and `verify()`'s own docstring already records that homogeneity holds for
any symmetric per-group scheme and therefore discriminates nothing.

The residual comes from the activation scale being stored in BF16, not float32. Measured on CPU
over 200k random amax values, `7 * bfloat16(amax / 7)` never reproduces `amax`: mean relative
error 0.0014, max 0.0039. So treat the recovery as good to ~1e-3 relative, not to zero.

**[GPU]** Two caveats on that number, both unclosed: it assumes nunchaku stores the INT4 `oscales` in BF16
(read from `ops/quantize.py` and `models/linear.py`, not executed), and the test that would catch
it -- `test_identity_probe_is_exact` -- runs `FakeW4A4` in float32, where the error structurally
cannot appear. Closing it needs a GPU: quantize a known BF16 weight with nunchaku's own quantizer
and compare the recovery against an fp32 dequantization.

`--verify` checks the conditions the recovery assumes -- bias suppression, kernel determinism,
finiteness -- and prints a diagnostic table it does **not** gate on. Replaying the layer on
random input cannot be a gate: measured on this kernel, a correct `feed_forward.net.2` scores
cosine 0.704 while a matrix that is 20% wrong scores 0.976, because the activation-quantisation
error on random input is larger than the weight error being looked for. Whether the
reconstruction is any good is a question only `--reference` can answer.

Needs CUDA. The recovery *is* the Nunchaku kernel running -- an Ampere card or newer, the same
one that runs the model.

    python tools/svdq_to_bf16.py --input svdq-int4_r32-beyond-reality-zimage-v2.safetensors \\
        --output beyond-reality-zimage-v2_bf16.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
COPY_CHUNK = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="SVDQuant checkpoint (file or model name)")
    parser.add_argument("--output", required=True, help="where to write the BF16 result")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify", type=int, default=8,
                        help="how many layers get the full check (bias suppression, kernel "
                             "determinism, finiteness) and the per-layer diagnostic table. 0 "
                             "disables. Bias suppression is checked on every layer either way, "
                             "because it costs one forward on zeros and a failure there is "
                             "wrong everywhere at once.")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke mode: recover only the first N layers, print the "
                             "diagnostics, and write NOTHING. It cannot write, because the "
                             "layers past N would keep their raw SVDQuant tensors while the "
                             "metadata declared the whole file dequantized.")
    parser.add_argument("--keep-fused", action="store_true",
                        help="write the kernel's fused layers as they are, instead of splitting "
                             "them back into to_q/to_k/to_v and w3/w1. The fused form is what "
                             "the kernel holds, but ComfyUI's Z-Image loader wants the split "
                             "one and dies with KeyError on to_k otherwise -- so splitting is "
                             "the default and this flag is for inspecting the raw recovery.")
    parser.add_argument("--reference", help=(
        "directory or safetensors holding the ORIGINAL BF16 weights, when the publisher shipped "
        "them beside the quantised file. This is the ONLY check here with ground truth. "
        "--verify establishes that the probe ran under the right conditions; it has no oracle "
        "for whether the matrix is right, because the matrix IS the kernel's output. Comparing "
        "against the original says both that the recovery worked and how much SVDQuant lost."))
    return parser.parse_args()


def load_reference(path: Path) -> dict:
    """Load original weights from a file or a sharded diffusers directory."""
    from safetensors.torch import load_file

    shards = sorted(path.glob("**/*.safetensors")) if path.is_dir() else [path]
    if not shards:
        raise SystemExit(f"no .safetensors under {path}")
    merged: dict = {}
    for shard in shards:
        merged.update(load_file(str(shard)))
    return merged


def match_reference(reference: dict, stem: str) -> torch.Tensor | None:
    """Find the original weight for a quantised layer.

    The quantised file and the diffusers checkpoint usually agree on module paths, but not
    always on a prefix. Try the direct name first, then any key that ends with it -- and refuse
    to guess when more than one key would match, because a silent mis-pairing here reports a
    quantisation error that belongs to a different layer.
    """
    direct = f"{stem}.weight"
    if direct in reference:
        return reference[direct]
    candidates = [k for k in reference if k.endswith(f".{direct}") or k == direct]
    if len(candidates) == 1:
        return reference[candidates[0]]

    # The quantised file fuses layers the diffusers checkpoint keeps apart, so matching on name
    # alone silently skips most of the model -- on Beyond_Reality it compared 5 layers out of 20,
    # every one of them `to_out`, and reported that as if it covered the network. Verified
    # against the shapes in both files:
    #     to_qkv                  (11520, 3840) == to_q + to_k + to_v, 3 x (3840, 3840)
    #     feed_forward.net.0.proj (20480, 3840) == w1 + w3,             2 x (10240, 3840)
    #     feed_forward.net.2      (3840, 10240) == w2
    # These names are Lumina/NextDiT (Z-Image). Another architecture needs its own entries; an
    # unknown fusion returns None and is reported as unmatched rather than guessed at.
    fusions = {
        "attention.to_qkv": ("attention.to_q", "attention.to_k", "attention.to_v"),
        # w3 first, then w1 -- not the other way round. Measured, not assumed: the natural
        # w1+w3 order gives relative error 1.4199 against the original while w3+w1 gives 0.0995,
        # and splitting the recovered matrix in half pins it down (upper half matches w3 at
        # 0.099, lower half matches w1 at 0.1001).
        "feed_forward.net.0.proj": ("feed_forward.w3", "feed_forward.w1"),
        "feed_forward.net.2": ("feed_forward.w2",),
    }
    for suffix, parts in fusions.items():
        if not stem.endswith(suffix):
            continue
        prefix = stem[: -len(suffix)]
        keys = [f"{prefix}{part}.weight" for part in parts]
        if all(k in reference for k in keys):
            return torch.cat([reference[k] for k in keys], dim=0)
        return None
    return None


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return header, 8 + length


def resolve(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    for root in (PORTABLE_ROOT / "ComfyUI" / "models" / "diffusion_models",
                 Path("D:/ComfyUI-Models/diffusion_models")):
        if (root / name).is_file():
            return root / name
    raise SystemExit(f"{name} not found")


def recover_weight(tensors: dict, stem: str, device: str,
                   dtype: torch.dtype) -> tuple[object, torch.Tensor, float]:
    """The effective weight of one quantised layer, shape (out_features, in_features).

    Returns the layer as well, so `verify` can question the same object rather than rebuilding
    it, and the zero-input response, which is the evidence that the bias was really suppressed.
    """
    from nunchaku.models.linear import SVDQW4A4Linear

    qweight = tensors[f"{stem}.qweight"]
    out_features, half_in = qweight.shape
    in_features = half_in * 2
    rank = tensors[f"{stem}.proj_down"].shape[1]
    has_bias = f"{stem}.bias" in tensors

    layer = SVDQW4A4Linear(in_features, out_features, rank=rank, bias=has_bias,
                           precision="int4", torch_dtype=dtype, device=device)
    with torch.no_grad():
        layer.qweight.copy_(qweight.to(device))
        layer.wscales.copy_(tensors[f"{stem}.wscales"].to(device))
        layer.smooth_factor.copy_(tensors[f"{stem}.smooth_factor"].to(device))
        layer.smooth_factor_orig.copy_(tensors[f"{stem}.smooth_factor_orig"].to(device))
        layer.proj_down.copy_(tensors[f"{stem}.proj_down"].to(device))
        layer.proj_up.copy_(tensors[f"{stem}.proj_up"].to(device))
        if has_bias:
            layer.bias.copy_(tensors[f"{stem}.bias"].to(device))

        # The bias is written out separately and must not ride along in the probe: forward(I)
        # would return W.T + bias on every row.
        saved_bias = None
        if layer.bias is not None:
            saved_bias = layer.bias.detach().clone()
            layer.bias.zero_()

        # (1, in, in), not (in, in): SVDQW4A4Linear.forward unpacks `batch_size, seq_len,
        # channels = x.shape` and a 2-D probe dies with "not enough values to unpack".
        eye = torch.eye(in_features, dtype=dtype, device=device).unsqueeze(0)
        recovered = layer(eye)[0].T.contiguous()

        # With the bias suppressed the layer must map zero to zero. If it does not, the bias
        # was not in fact suppressed and every row of `recovered` is offset by it -- a failure
        # that produces a model which loads, runs, and is quietly wrong everywhere. Measured
        # here, while the bias is still zeroed, rather than reasoned about above.
        zeros = torch.zeros(1, 1, in_features, dtype=dtype, device=device)
        zero_leak = layer(zeros).abs().max().item()

        if saved_bias is not None:
            layer.bias.copy_(saved_bias)
    return layer, recovered, zero_leak


def split_fused(stem: str, weight: torch.Tensor) -> dict:
    """Split one recovered matrix back into the names a normal loader expects.

    The kernel keeps q/k/v in one matrix and the two MLP gates in another, so the raw recovery
    fails ComfyUI with `KeyError: 'noise_refiner.0.attention.to_k.weight'`. The cuts are fixed by
    the shapes; the only thing that is not obvious is the MLP order, and that was measured rather
    than guessed -- `w1+w3` scores relative error 1.4199 against the published BF16 while `w3+w1`
    scores 0.0995, and halving the recovered matrix pins each half to its own tensor.

    Validated at full scale on Beyond_Reality Z-Image v2: 521 tensors produced, 0 missing, 0
    extra, 0 with a wrong shape against the publisher's own file, byte size identical.

    Returns a dict of {key: tensor}. A stem this does not recognise passes through unsplit --
    silently writing a guess for an unknown fusion is how a file gets produced that loads and is
    wrong.
    """
    if stem.endswith("attention.to_qkv"):
        base = stem[: -len("to_qkv")]
        if weight.shape[0] % 3 == 0:
            return {f"{base}{n}.weight": c.contiguous()
                    for n, c in zip(("to_q", "to_k", "to_v"), weight.chunk(3, dim=0))}
    elif stem.endswith("feed_forward.net.0.proj"):
        base = stem[: -len("net.0.proj")]
        if weight.shape[0] % 2 == 0:
            top, bottom = weight.chunk(2, dim=0)
            return {f"{base}w3.weight": top.contiguous(),
                    f"{base}w1.weight": bottom.contiguous()}
    elif stem.endswith("feed_forward.net.2"):
        return {stem[: -len("net.2")] + "w2.weight": weight}
    return {f"{stem}.weight": weight}


def _in_out(header: dict, stem: str) -> tuple[int, int] | None:
    """`(out_features, in_features)` for a quantised stem, from its low-rank pair.

    Not from `qweight`: that is the INT4 container and its second axis is half of `in_features`.
    `proj_up` is `[out, rank]` and `proj_down` is `[in, rank]`, both stored uncompressed.
    """
    down, up = header.get(f"{stem}.proj_down"), header.get(f"{stem}.proj_up")
    if not down or not up:
        return None
    return up["shape"][0], down["shape"][0]


def check_fusion(header: dict, stems: list[str], config: dict) -> list[str]:
    """Refuse, before a single tensor is read, the fusions `split_fused` cannot see from a shape.

    `split_fused` decides by name and by divisibility. Divisibility is not the property that
    matters -- three of the four ways this can go wrong leave the divisibility check green and
    produce a file that loads, or half-loads, and is wrong. Each check below refuses one of them,
    and each is here because it was **measured** on the SVDQuant checkpoints on this machine
    (2026-08-21, headers only, four files: z-image-turbo, beyond-reality-zimage-v2, qwen-image,
    flux.1-dev):

    * **A fused stem that carries a bias.** The weight is split into `to_q/to_k/to_v.weight`
      while `to_qkv.bias` rides through the passthrough under the *fused* name, so the output has
      three weights and a bias none of them matches. Latent where `split_fused` matches today
      (the 102 matching stems in each Z-Image file have zero bias) and **active the moment the
      pattern is widened**: Qwen-Image has 300 stems one prefix away from matching (`attn.to_qkv`
      rather than `attention.to_qkv`) and **every one of them has a bias**.

    * **`to_qkv` where `out != 3 * in`.** Equal thirds is what `chunk(3)` assumes; GQA breaks it
      while leaving `out % 3 == 0` perfectly possible. Measured: `out == 3 * in` on all 94 fused
      attention stems across the four files, so refusing on inequality costs nothing today.

    * **`n_kv_heads != n_heads` in the model's own config.** The direct statement of the same
      thing, when the publisher wrote it down. Z-Image carries both keys (30 and 30 here).

    * **`net.0.proj` that is a single projection, not a gate pair.** The decisive evidence is the
      sibling `net.2` (the `w2` that consumes the gate output): if the projection is a gate pair,
      `w2` takes half of it. Measured -- Z-Image `net.2` takes 10240 of `net.0.proj`'s 20480, a
      gate pair; **Qwen-Image `net.2` takes 12288 of 12288**, so Qwen's `net.0.proj` is a single
      `gelu` projection and cutting it in half would write two tensors nothing consumes. That is
      the scenario `AUDITORIA_2026-08-18.md` called constructed; it is sitting in a real file.

    Returns one string per problem, empty when there is nothing to refuse. **This is a
    header-level check: shapes and names only.** It does not run the kernel, does not look at a
    single weight value, and therefore cannot tell a correct split from a plausible one -- it
    only refuses the cases that are provably wrong before any work starts.
    """
    problems: list[str] = []
    n_heads, n_kv = config.get("n_heads"), config.get("n_kv_heads")
    for stem in stems:
        split_keys = split_fused(stem, torch.empty(6, 1))
        if set(split_keys) == {f"{stem}.weight"}:
            continue                                  # passthrough: nothing is being cut
        if f"{stem}.bias" in header:
            problems.append(
                f"{stem}: the weight is renamed to {sorted(split_keys)} but "
                f"{stem}.bias passes through under the fused name, so no split weight would "
                f"match it")
        shape = _in_out(header, stem)
        if stem.endswith("attention.to_qkv"):
            if shape and shape[0] != 3 * shape[1]:
                problems.append(
                    f"{stem}: out={shape[0]} is not 3*in={3 * shape[1]}, so q/k/v are not equal "
                    f"thirds (GQA?) and chunk(3) would cut in the wrong places")
            if n_heads is not None and n_kv is not None and n_heads != n_kv:
                problems.append(
                    f"{stem}: the model config says n_heads={n_heads} but n_kv_heads={n_kv}, "
                    f"so k and v are narrower than q and this is not a three-equal-way fusion")
        elif stem.endswith("feed_forward.net.0.proj"):
            sibling = _in_out(header, stem[: -len("net.0.proj")] + "net.2")
            if shape and sibling and sibling[1] != shape[0] // 2:
                problems.append(
                    f"{stem}: out={shape[0]}, but the sibling net.2 consumes {sibling[1]}, not "
                    f"{shape[0] // 2}. A gate pair feeds w2 half of it; this one is a single "
                    f"projection and splitting it in two would write tensors nothing reads")
    return problems


def verify(layer, weight: torch.Tensor, device: str, dtype: torch.dtype) -> dict:
    """Diagnostic scores for one recovered layer. **Reported, never gated.** Read why.

    This was a gate twice, and was wrong twice. Both attempts are recorded here because the
    numbers are the argument.

    First version: require `layer(x) ~= x @ W.T` to 2e-2 on random input. Impossible. W4A4
    quantises the *activation*, so `layer` is not linear in its input and no BF16 matrix
    reproduces it. It measured 0.0988 and rejected correct reconstructions.

    Second version: keep the comparison but gate on direction -- cosine >= 0.99, with a
    shuffled-matrix control to prove the margin. Calibrated against a stand-in layer that
    quantised its activation but had no smoothing and no low-rank branch. On the *real* kernel
    (Z-Image INT4 r32, 14 layers sampled across the network) that calibration is fiction:

        correct reconstruction      cos 0.704 .. 0.998    relL2 0.058 .. 0.894
        rows shuffled               cos 0.004 max         relL2 ratio 0.56
        correct x 1.5               cos 0.995 max         relL2 ratio 0.65
        correct + 20% noise         cos 0.976 max         relL2 ratio 0.963

    A correct `feed_forward.net.2` on a hard probe scores 0.704. A matrix that is 20% wrong
    scores 0.976. The wrong matrix outscores the right one, so no threshold on this measurement
    separates them -- the activation-quantisation error on random input is larger than the
    weight error being looked for. It is the same category error as an FBCache threshold that
    does not port between architectures, and for the same reason: the number is a property of
    the layer, not of the thing being tested.

    Homogeneity was tried too (`layer(cI)/c == layer(I)`): exactly 0 on every layer, and also
    ~0 on random input, so it distinguishes nothing. Symmetric per-group quantisation is
    homogeneous for any input.

    What survives as an actual check is in `check_recovery`, and `--reference` for real
    correctness. What this function returns is still worth printing -- it says how much the
    4-bit activation path costs on each layer, and the shuffled control says what no information
    looks like -- but it is a description, not a verdict.
    """
    with torch.no_grad():
        probe = torch.randn(1, 64, layer.in_features, dtype=dtype, device=device)
        reference = layer(probe)[0]
        probe = probe[0]
        if layer.bias is not None:
            reference = reference - layer.bias

        def score(matrix: torch.Tensor) -> tuple[float, float]:
            replayed = (probe @ matrix.T).float()
            ref = reference.float()
            rel = ((ref - replayed).norm() / ref.norm().clamp(min=1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(
                ref.flatten().unsqueeze(0), replayed.flatten().unsqueeze(0)).item()
            return rel, cos

        rel, cos = score(weight)
        # Fixed seed: the control is a reference point printed next to every layer, and one that
        # moved run to run would be useless for comparing two runs of this tool.
        order = torch.randperm(weight.shape[0],
                               generator=torch.Generator().manual_seed(0)).to(weight.device)
        control_rel, control_cos = score(weight[order])
    return {"rel": rel, "cos": cos, "control_rel": control_rel, "control_cos": control_cos}


def check_recovery(layer, weight: torch.Tensor, device: str, dtype: torch.dtype,
                   zero_leak: float) -> str | None:
    """The checks that actually decide. Returns a reason to stop, or None.

    Short list, because the honest list is short. `recovered` *is* `layer(I)`, so there is no
    independent oracle inside this tool to compare it against -- only `--reference` has one. What
    can be established without ground truth is that the probe was run under the conditions the
    recovery assumes:

      bias         with the bias suppressed the layer must map zero to zero, or every recovered
                   row is offset by it. Measured 0.0 on every layer of every checkpoint tried,
                   and a nonzero here means the recovery is wrong everywhere at once.
      determinism  two identical identity probes must return bit-identical results. A kernel
                   that does not is one whose output cannot be treated as "the weight", and a
                   flaky reduction would show up as a model that is subtly wrong in a way no
                   later check would attribute to this step.
      magnitude    an all-zero or non-finite recovered matrix is a failed kernel launch that
                   raised nothing. Cheap to rule out, and otherwise it writes a dead file.

    Deliberately not here: anything derived from replaying the layer on random input. See
    `verify` for the measurements showing that a matrix 20% wrong outscores a correct one there.
    """
    if zero_leak > 1e-3:
        return (f"the layer returned {zero_leak:.4g} for a zero input with its bias suppressed; "
                f"every recovered row would carry that offset")
    if not torch.isfinite(weight).all():
        return "the recovered matrix contains inf or nan"
    if weight.abs().max().item() == 0.0:
        return "the recovered matrix is all zeros; the kernel produced nothing"

    with torch.no_grad():
        saved = None
        if layer.bias is not None:
            saved = layer.bias.detach().clone()
            layer.bias.zero_()
        eye = torch.eye(layer.in_features, dtype=dtype, device=device).unsqueeze(0)
        again = layer(eye)[0].T.contiguous()
        if saved is not None:
            layer.bias.copy_(saved)
    if not torch.equal(again, weight):
        drift = (again.float() - weight.float()).abs().max().item()
        return (f"two identical identity probes disagreed by {drift:.4g}; the kernel is not "
                f"deterministic, so its output cannot be taken as the layer's weight")
    return None


def write_checkpoint(src: Path, data_start: int, out: Path, partial: Path,
                     entries: list, blob: bytes, planned: int) -> None:
    """Stream the planned file out, and refuse to hand over anything that is not the plan.

    This was the only writer in `tools/` without the contract the other six share. Find them with

        grep -n '"xb"' tools/*.py

    -- exactly one hit in each of `quant_w4a4.py`, `quant_w4a8.py`, `quant_int8.py`,
    `quant_w4a4_smooth.py`, `quant_mixed.py` and `to_native.py`, and several in this file, of
    which only the `open(partial, "xb")` below is code. This paragraph
    carried the six line ranges instead, and `quant_mixed.py:637-657` was ALREADY WRONG in the
    commit that introduced it: the same commit added 213 lines to that file and moved its writer
    to :798. Re-checked 2026-08-22, EXECUTED: four of the six had drifted again within one
    session, three of them between two greps ten minutes apart, because sibling agents were
    editing those files at the time. A line number in a comment is a claim with an expiry date
    nobody can see; a symbol plus the grep that finds it does not rot.

    What that contract is: the partial opened `"wb"` here (clobbering a crashed run's leftover),
    bytes written were never compared against bytes planned, nothing was `fsync`ed, and there was
    no `try/finally`, so a failure left the partial behind for the next run to overwrite.

    That gap is worse here than in any of the six, for two reasons. A safetensors header is
    self-describing, so a file short by one tensor still parses and still *loads* -- the tail
    tensors come back as garbage and the model produces a wrong image instead of an exception.
    And this is the one converter whose output has no BF16 original to fall back on; that is the
    whole reason the tool exists (see the module docstring).

    Hence: `"xb"`, so the partial must not already exist; `written == planned` checked before
    the file is allowed to become `out`; `flush` + `fsync` before `os.replace`; and the partial
    unlinked in `finally` on every exit path, success or not.

    Provenance: EXECUTED, but not on a real checkpoint. On a hand-built temp-dir safetensors
    (two copied tensors plus one written one, 44 planned bytes) an honest write round-trips
    through `load_file`; a payload truncated after planning raises
    `RuntimeError(length mismatch: wrote 28, planned 44)` and leaves neither `out` nor
    `.partial`; a pre-existing `.partial` raises `FileExistsError`. Never run at full scale
    since this change -- that needs CUDA and an 11 GiB source.
    """
    try:
        with open(partial, "xb") as dst, open(src, "rb") as source:
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            body_start = dst.tell()
            for key, _, (kind, payload) in entries:
                if kind == "copy":
                    start, end = payload["data_offsets"]
                    source.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        chunk = source.read(min(COPY_CHUNK, remaining))
                        if not chunk:
                            raise RuntimeError(f"source ended early while copying {key}")
                        dst.write(chunk)
                        remaining -= len(chunk)
                else:
                    dst.write(payload.contiguous().view(torch.uint8).numpy().tobytes())
            written = dst.tell() - body_start
            if written != planned:
                raise RuntimeError(f"length mismatch: wrote {written}, planned {planned}")
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(partial, out)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device: the recovery runs the Nunchaku kernel, which needs one")
        return 1

    from safetensors.torch import load_file

    src = resolve(args.input)
    out = Path(args.output)
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists():
        print(f"{out} already exists; refusing to overwrite")
        return 1
    # Refused here, not at the write, which is hours of kernel time later. The other six
    # converters all refuse a stale partial before starting work; this one used to open it
    # "wb" at the very end and clobber a crashed run's leftover without a word.
    if partial.exists():
        print(f"refusing to overwrite stale partial output: {partial}")
        return 1

    header, data_start = read_header(src)
    metadata = header.pop("__metadata__", None)
    stems = sorted({k.rsplit(".", 1)[0] for k in header if k.endswith(".qweight")})
    if not stems:
        print("no .qweight tensors: this does not look like an SVDQuant checkpoint")
        return 1
    if args.limit:
        stems = stems[:args.limit]
    print(f"{src.name}: {len(header)} tensors, {len(stems)} quantised layers")

    quant_cfg = json.loads(metadata.get("quantization_config", "{}")) if metadata else {}
    if quant_cfg.get("weight", {}).get("dtype") not in (None, "int4"):
        print(f"only int4 is handled; this file is {quant_cfg['weight']['dtype']} "
              f"(nvfp4 needs Blackwell and a different unpack path)")
        return 1
    print(f"rank {quant_cfg.get('rank')}, group_size "
          f"{quant_cfg.get('weight', {}).get('group_size')}")

    # Refuse before touching a tensor, not after writing 11 GiB. `--keep-fused` skips it because
    # nothing is being split in that mode, so none of the three failures can happen.
    if not args.keep_fused:
        config = json.loads(metadata.get("config", "{}")) if metadata else {}
        problems = check_fusion(header, stems, config)
        if problems:
            print(f"\nrefusing to convert: {len(problems)} fusion(s) this tool cannot split "
                  f"correctly in this file.")
            for problem in problems:
                print(f"  {problem}")
            print("\nNothing was written. Either pass --keep-fused to write the fused weights "
                  "under their fused names, or teach split_fused this architecture and re-run.")
            return 1

    # BF16 for the recovery, and only for the recovery. Everything that is not part of a
    # quantised layer -- norms included -- is copied byte-for-byte further down with the source's
    # own `info["dtype"]`, so a checkpoint whose norms are F32 keeps them F32. Measured
    # 2026-08-21 on four SVDQuant files here (z-image-turbo, beyond-reality-zimage-v2,
    # qwen-image, flux.1-dev): every passthrough tensor is BF16 in all four, and the copy path
    # never converts. `AUDITORIA_2026-08-18.md` listed "dtype BF16 hardcoded for the norms" as an
    # open low-confidence finding; the norms are not the ones at risk here. What IS hardcoded is
    # the dtype the Nunchaku layer is built with, below.
    dtype = torch.bfloat16
    print("loading source into RAM", flush=True)
    tensors = load_file(str(src))

    # Anything that is not part of a quantised layer is carried over untouched. The six suffixes
    # below are consumed by the recovery and must not appear in the output; `bias` is kept,
    # because the reconstructed layer still needs it.
    consumed = {"qweight", "wscales", "smooth_factor", "smooth_factor_orig",
                "proj_down", "proj_up"}
    stem_set = set(stems)
    passthrough = [k for k in header
                   if not (k.rsplit(".", 1)[0] in stem_set and k.rsplit(".", 1)[-1] in consumed)]

    reference = None
    if args.reference:
        print(f"loading reference weights from {args.reference}", flush=True)
        reference = load_reference(Path(args.reference))
        print(f"  {len(reference)} tensors", flush=True)

    print(f"recovering {len(stems)} layers on {args.device}", flush=True)
    recovered: dict[str, torch.Tensor] = {}
    errors: list[tuple[str, float]] = []
    ref_errors: list[tuple[str, float]] = []
    unmatched: list[str] = []
    started = time.perf_counter()
    for index, stem in enumerate(stems):
        layer, weight, zero_leak = recover_weight(tensors, stem, args.device, dtype)
        # The cheap decisive checks run on the sampled layers; the two that cost nothing (bias,
        # finiteness) would run everywhere if the determinism re-probe did not double the work.
        if index < args.verify:
            why = check_recovery(layer, weight, args.device, dtype, zero_leak)
            if why:
                print(f"\nRECOVERY FAILED on {stem}: {why}.")
                print("not writing anything")
                return 1
            result = verify(layer, weight, args.device, dtype)
            result["zero_leak"] = zero_leak
            errors.append((stem, result))
        # `not (zero_leak <= 1e-3)`, not `zero_leak > 1e-3`. In Python `float('nan') > 1e-3` is
        # False, so the old form let a NaN through -- and for every layer at or past --verify
        # (128 of the 136 in Z-Image at the default of 8) this is the *only* check that runs;
        # `check_recovery`, which calls torch.isfinite, only runs on the sample. A recovery that
        # went to NaN wrote 11.2 GiB and exited 0.
        elif not (zero_leak <= 1e-3) or not torch.isfinite(weight).all():
            reason = ("produced a non-finite recovered matrix" if not torch.isfinite(weight).all()
                      else f"returned {zero_leak:.4g} for a zero input with its bias suppressed; "
                           "every recovered row would carry that offset")
            print(f"\nRECOVERY FAILED on {stem}: the layer {reason}.")
            print("not writing anything")
            return 1
        if reference is not None:
            original = match_reference(reference, stem)
            if original is None:
                unmatched.append(stem)
            else:
                original = original.to(weight.device, weight.dtype)
                if original.shape != weight.shape:
                    unmatched.append(f"{stem} (shape {tuple(original.shape)} vs "
                                     f"{tuple(weight.shape)})")
                else:
                    err = ((original.float() - weight.float()).norm()
                           / original.float().norm().clamp(min=1e-12)).item()
                    ref_errors.append((stem, err))
        if args.keep_fused:
            recovered[f"{stem}.weight"] = weight.to("cpu")
        else:
            for key, part in split_fused(stem, weight).items():
                recovered[key] = part.to("cpu")
        del layer, weight
        if (index + 1) % 25 == 0 or index + 1 == len(stems):
            done = index + 1
            rate = done / (time.perf_counter() - started)
            print(f"  {done}/{len(stems)}  {rate:.1f} layers/s", flush=True)
    torch.cuda.empty_cache()

    if errors:
        print(f"\nchecked {len(errors)} layers: bias suppression, kernel determinism, "
              f"finiteness -- all passed.")
        print("\nActivation-quantisation cost per layer, on random input. **Not a verdict.**")
        print(f"  {'layer':<44}{'cos':>8}{'relL2':>9}{'cos(wrong)':>12}{'relL2(wrong)':>14}")
        for stem, r in errors:
            print(f"  {stem[-44:]:<44}{r['cos']:>8.4f}{r['rel']:>9.4f}"
                  f"{r['control_cos']:>12.4f}{r['control_rel']:>14.4f}")
        rels = sorted(r["rel"] for _, r in errors)
        print(f"  relL2 spread {rels[0]:.3f} .. {rels[-1]:.3f} across these layers.")
        print("  The last two columns are the same measurement against a shuffled copy of the "
              "recovered\n  matrix -- what having no information looks like. Do not turn any of "
              "this into a\n  threshold: measured on Z-Image INT4, a matrix 20% wrong scores "
              "cosine 0.976 while a\n  correct feed_forward.net.2 scores 0.704, so the wrong "
              "matrix outranks the right one.\n  Only --reference can say whether the "
              "reconstruction is good.")

    if ref_errors:
        values = sorted(e for _, e in ref_errors)
        worst = max(ref_errors, key=lambda x: x[1])
        median = values[len(values) // 2]
        print(f"\nvs original BF16, over {len(ref_errors)} layers:")
        print(f"  median relative error : {median:.4f}")
        print(f"  best                  : {values[0]:.4f}")
        print(f"  worst                 : {worst[1]:.4f}  ({worst[0]})")
        print("This is the distance between the published INT4 checkpoint and its BF16 "
              "source."
              "\nIt is not this tool's error: the identity probe reads the weights the "
              "kernel itself holds. A nonzero number here is what 4-bit cost.")
    if unmatched:
        print(f"\n{len(unmatched)} layer(s) had no unambiguous match in the reference "
              f"(not compared):")
        for name in unmatched[:5]:
            print(f"  {name}")

    # --limit used to write a FULL output file. The layers past N never entered `stem_set`, so
    # their raw SVDQuant tensors (qweight, wscales, proj_up/proj_down, the smooth factors) rode
    # through the passthrough list untouched while the metadata declared the whole file
    # dequantized -- a `--limit 4` smoke run produced something that looks like a finished
    # checkpoint and is not one.
    #
    # It refuses to write rather than writing a file whose metadata marks it partial: nothing
    # would read that mark. There is no verifier for this tool's output -- `verify_w4a4.py` and
    # its siblings check W4A4 checkpoints, not dequantized BF16 ones -- and a partial-flag that
    # no gate enforces is the exact shape of claim this bench keeps getting burned by. A mode
    # that writes nothing cannot be misread.
    #
    # **What this cost, stated because the change that made it did not state it.** `--limit N`
    # was the only cheap route to `write_checkpoint`: four layers, seconds of kernel time, and
    # the writer ran. Returning here means the writer is now reachable ONLY from a full run --
    # all 136 Z-Image layers off an 11 GiB source, on CUDA. And `tools/test_svdq_verify.py` has
    # no case for `write_checkpoint` or `check_fusion` at all (grep, 2026-08-22, executed: zero
    # hits for either symbol), so the contract added above -- "xb", written == planned, fsync,
    # the finally -- rests on one ad-hoc temp-dir run recorded in that function's docstring and
    # on nothing that re-runs. That is a fair trade against writing a checkpoint that is not one,
    # but it is a trade, and the coverage it spent has to be bought back somewhere: a CPU case in
    # test_svdq_verify.py that builds a small safetensors, calls `write_checkpoint` directly, and
    # asserts the truncated write raises and leaves neither `out` nor `.partial`. It needs no GPU
    # and does not exist yet.
    if args.limit:
        print(f"\n--limit {args.limit}: smoke mode, nothing written. {len(recovered)} tensor(s) "
              f"were recovered and discarded.")
        print(f"Re-run without --limit to write {out}.")
        return 0

    # Header first, then stream: the recovered weights alone are several times the size of the
    # input, and building the whole file in memory before writing is how a conversion turns into
    # an OOM on a machine that shares RAM with other work.
    entries: list[tuple[str, dict, object]] = []
    offset = 0
    for key in passthrough:
        info = header[key]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        entries.append((key, {"dtype": info["dtype"], "shape": info["shape"],
                              "data_offsets": [offset, offset + size]}, ("copy", info)))
        offset += size
    for key, tensor in recovered.items():
        size = tensor.numel() * tensor.element_size()
        entries.append((key, {"dtype": DTYPE_NAMES[tensor.dtype], "shape": list(tensor.shape),
                              "data_offsets": [offset, offset + size]}, ("write", tensor)))
        offset += size

    out_header = {key: info for key, info, _ in entries}
    if metadata:
        kept = dict(metadata)
        # The quantisation_config described a file that no longer exists in this form. Leaving it
        # would make a BF16 checkpoint claim to be INT4, and a loader may believe it.
        kept.pop("quantization_config", None)
        kept["dequantized_from"] = src.name
        kept["dequantized_note"] = ("weights recovered from SVDQuant by identity probe; "
                                    "quality is that of the INT4 source, not of any BF16 original")
        out_header["__metadata__"] = kept

    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)

    print(f"writing {out.name} ({(offset + len(blob) + 8) / 2**30:.2f} GiB)", flush=True)
    write_checkpoint(src, data_start, out, partial, entries, blob, offset)
    print(f"done: {out}")
    print("reminder: this carries INT4 quality at BF16 size. It is only worth keeping for a "
          "model you do not have a BF16 original of.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
