# 12 - quant_mixed matches calibration to checkpoint by basename, and downcasts the reservoir it was told not to

Type: task
Status: ready-for-agent
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
