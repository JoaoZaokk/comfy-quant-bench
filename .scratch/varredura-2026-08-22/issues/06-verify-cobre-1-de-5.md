# 06 - verify_w4a4 smoke-tests a rotation the file may not use, and covers one of five output formats

Type: task
Status: ready-for-agent
Blocked by: 05
Severity: medium
Provenance: TRACED, then EXECUTED 2026-08-22 on the 3090

## Problem

**a) Hardcoded groupsize.** `verify_w4a4.py:255-256` reads per-layer metadata into `layers`.
`kernel_smoke` at `:223-226` then hardcodes `convrot_groupsize: 256, quant_group_size: 64` and never
consults it. A checkpoint written by `quant_w4a4.py --convrot-groupsize 64` -- a supported, documented
flag whose help text at `:73-76` says "a file converted at 64 is executed at 64" -- is smoke-tested at
256. The RMSE ceiling is 0.9 (`:28`), deliberately loose. Two outcomes, both bad: a correct file fails,
or the mismatch lands under 0.9 and the tool prints PASS for a comparison it never performed.

**b) One format of five.** `verify_w4a4.py:106-109` rejects any layer whose `format` is not
`convrot_w4a4`. `quant_mixed.py:590-595` writes `asym_w4a8_int8` alongside it, so a mixed checkpoint
fails structural verification outright. `quant_w4a8.py` and `quant_int8.py` outputs have no verifier
at all -- grep for `int8_tensorwise` and `asym_w4a8_int8` outside the converters hits only
`_native_probe.py` and `check_w4a8.py`, and `check_w4a8.py:1-20` is a kernel-correctness study on
synthetic tensors, not a checkpoint verifier.

**The byte-identical-preserved-tensor check at `:178-205` -- arguably the single most valuable
guarantee this project has -- is available to one of five output formats.**

**c) No caveat block.** See ticket 02.

## The deepening, not just the fix

The format-specific part of verification is *only* `validate_structure`'s per-layer expectations
(`:106-132`): which auxiliary tensors, which dtypes, which shape relation. Everything else --
`read_header`, `validate_preserved_bytes`, the extra-key walk at `:145-157`, backend resolution, the
smoke -- is format-agnostic.

A `Format` adapter with `expected_tensors(layer, config)` and `smoke_kwargs(layer, config)` -- three
small implementations -- makes byte-identity and the caveat block apply to all five, and makes the
groupsize come from the file instead of from a literal.

## Closing criterion (written before the fix)

Closed when:

1. `kernel_smoke` reads `convrot_groupsize` and `quant_group_size` from the layer's own metadata, and
   raises rather than defaulting if either is absent;
2. `verify_w4a4.py` (renamed or not) structurally verifies a `quant_mixed` output, a `quant_w4a8`
   output and a `quant_int8` output, with the byte-identical preserved-tensor comparison running for
   all of them;
3. one existing converted checkpoint of each format on disk passes, and one deliberately corrupted
   copy of each fails -- the corruption being a single flipped byte in a *preserved* tensor, which is
   what that check exists to catch;
4. every run ends with a statement of what it did not cover.

Item 3 needs no GPU for the structural half. The `--kernel-smoke` half does.

## Answer -- EXECUTED 2026-08-22 on the RTX 3090, GPU lock held

All three parts measured. **(a) came out the better of the two ways it could go; (b) and (c) are
confirmed on real checkpoints, not synthetics.**

### (a) A groupsize mismatch FAILS loudly. It does not print a false PASS.

I wrote: *"Two outcomes, both bad: a correct file fails, or the mismatch lands under 0.9 and the
tool prints PASS for a comparison it never performed."* It is the first one:

    made at   run at   rel RMSE   verdict
         64       64     0.2137      PASS   <-- matched
         64      256     1.0230      FAIL   <-- MISMATCH
        256       64     1.0236      FAIL   <-- MISMATCH
        256      256     0.2136      PASS   <-- matched

So the cost is a **false alarm on a correct file**, not a false pass on a broken one. Severity
drops to low.

**But read the margin before relaxing.** The ceiling is 0.9 and the mismatch lands at 1.023 --
**14% of headroom.** Nothing about 0.9 was chosen with this in mind; `verify_w4a4.py:28` describes
it as a deliberately loose liveness signal. A future loosening to 1.1, which would look harmless,
flips this to the false-PASS case. **The fix is still worth doing, and the margin is the reason to
say so.**

### (b) CONFIRMED on a real mixed checkpoint, not a synthetic one

`zimage-v2-mixed.safetensors` (3.41 GB, 170 layers: **115 `convrot_w4a4` + 55 `asym_w4a8_int8`**,
both from `beyond-reality-zimage-v2_native.safetensors`). Run through `verify_w4a4.py`:

    ERROR: layers.27.feed_forward.w3: unexpected format 'asym_w4a8_int8'
    ERROR: layers.28.attention.out:   unexpected format 'asym_w4a8_int8'
    ... 55 of these

Every w4a8 layer rejected. **The byte-identical preserved-tensor comparison never runs on this
file** -- verification aborts at the structural stage. A real, on-disk, produced-by-our-own-tool
checkpoint has no verifier.

### (c) CONFIRMED. There is no caveat block.

Full output of `verify_w4a4.py zimage-v2-w4a4.safetensors --kernel-smoke`:

    Structural verification: PASS (170 ConvRot W4A4 layers)
    Source comparison: PASS
    Normal ComfyUI backend: comfy_kitchen.backends.cuda
    {
      "layer": "context_refiner.0.attention.out",
      "backend": "comfy_kitchen.backends.cuda.convrot_w4a4_linear",
      "output_dtype": "torch.bfloat16",
      "relative_rmse": 0.24021309614181519,
      "max_abs_error": 51.375
    }

It ends there. Two PASS lines and a JSON blob, no statement of scope. See ticket 02.

Incidental cross-check worth keeping: the real checkpoint's smoke RMSE is **0.2402** against my
synthetic **0.2136** at the same groupsize -- close enough that the synthetic was a fair stand-in,
which is itself worth knowing next time someone wants to test this without a 3 GB file.

### Closing criterion, unchanged

All four items stand. Item 3 (corrupt a preserved tensor and see it caught) is now the *only*
untested one, and it is the most important: on the mixed file that check does not run at all.
