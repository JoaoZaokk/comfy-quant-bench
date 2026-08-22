# 12 - quant_mixed matches calibration to checkpoint by basename, and downcasts the reservoir it was told not to

Type: task
Status: resolved
Blocked by: -
Severity: medium
Provenance: TRACED

## Problem

`tools/quant_mixed.py` decides, per layer, whether a weight gets 4-bit or 8-bit, by measuring the real
error on real activations. That decision is only as good as the provenance of the inputs.

- **`:344` validates the calibration blob by source BASENAME only.** Two different checkpoints with
  the same filename in different directories -- which this bench has, because outputs are written
  *beside* their source with a suffix -- match each other.
- **`:295` the `--analysis` provenance checks are all skipped when a key is missing.** A hand-edited or
  older analysis file with fewer keys passes every check by having nothing to check.
- **`:397` downcasts the bf16 activation reservoir to the checkpoint dtype before measuring.** The
  reservoir is bf16 *on purpose*: CLAUDE.md records that real Z-Image activations reach 344064, fp16
  caps at 65504, and an `inf` there makes every error `nan` -- which silently routes the worst layer in
  the model to the cheapest format. Downcasting at measurement time re-opens that door from the other
  end.
- `tools/calibrate_activations.py:11` -- **its own docstring still says the reservoir is fp16.** The
  exact statement the bug was about.

## Why this cluster is worth one ticket

Each item is individually low. Together they describe a tool whose output is a per-layer precision
assignment that nothing downstream can audit: there is no way, from the produced checkpoint, to tell
whether the analysis it was built from described that checkpoint.

## Closing criterion (written before the fix)

Closed when:

1. the calibration and analysis files carry a digest of the source header (not the whole file -- the
   header is enough to identify a checkpoint and costs nothing), and `quant_mixed` refuses on mismatch
   rather than matching on basename;
2. a missing provenance key is a refusal, not a skip;
3. the error measurement runs at bf16 or float32 and never at the checkpoint dtype, with an assertion
   and a comment naming the 344064-vs-65504 incident;
4. `calibrate_activations.py`'s docstring says bf16;
5. `nan`/`inf` in any per-layer error metric is a hard failure that names the layer, never a value that
   participates in a comparison.

Item 5 is the one that matters most and is the cheapest.

## Closed 2026-08-22, commit `13fbd8c` -- with the round's worst regression caught and reversed

Items 2, 3, 4 and 5 met and independently verified. Item 5 -- a `nan`/`inf` in any per-layer error
metric is a hard failure naming the layer -- is guarded at production *and* at ingestion *and*
again on the merged path before the decision loop.

### Item 1 shipped a guard WEAKER than the one it replaced. Measured.

The basename comparison was replaced by a sha256 of the safetensors header, on the stated
reasoning that *"the header alone identifies a checkpoint -- every tensor name, dtype, shape and
offset is in it"*. Every clause true; the conclusion false. The header describes the **layout**,
and two models of one architecture have identical layouts and different weights.

Over the 45 `.safetensors` in `ComfyUI/models/diffusion_models`, `ComfyUI/models/unet` and
`D:/ComfyUI-Models/diffusion_models` -- **four colliding groups covering nine files**, each group
also identical in byte size:

    e36743d8e80c9cef   beyond-reality-zimage-v2_native / z_image_de_turbo_v1_bf16 / z_image_turbo_bf16
    bced9dae1b9a4c0c   beyond-reality-zimage-v2_bf16 / beyond-reality-recovered-bf16
    c78d89e7215f3a98   void_pass2 / void_pass1
    a904e26816259491   wan2.2_i2v_high_noise_14B_fp8_scaled / wan2.2_i2v_low_noise_14B_fp8_scaled

The first group is *the three checkpoints `--foreign-analysis`'s own help text names as different
models*. The third is two passes of one conversion. The fourth is the two halves of a Wan pair.
Every basename differs -- so the check being replaced **refused** these pairings and the new one
**accepted** them, silently, on the model family this bench actually converts.

**Fixed.** The digest samples the body: 8 windows of 256 KiB spread across the data block, 2 MiB
total, still O(1) in model size. 45 files, 45 distinct digests, zero collisions. And `foreign` is
now `digest differs OR basename differs`, so the weaker signal can only ever add a refusal, never
remove one. Field renamed `source_identity_sha256` -- a field called `source_header_sha256` that
is not a header sha256 is how the next reader concludes the wrong thing.

A regression test was added, because the existing one could not see this: it varies the tensor
SHAPE, so the two files get different headers and a header-only digest separates them. The new
case builds two 4 KiB files with **byte-identical headers** and different weights.
