# 08 - One converter core: five converters re-implement the same eight-part contract and the parts diverge

Type: grilling
Status: ready-for-human
Blocked by: 05, 07
Severity: medium
Provenance: TRACED

## Question

Should the five quantizing converters plus `to_native.py` and `svdq_to_bf16.py` route their writing
through one module, with the per-format quantize step as the only plugged-in part?

This is a `grilling` ticket, not a `task`: it is the largest structural change proposed by the audit
and the owner should decide the shape before anyone types it.

## The contract, written out longhand six times

1. refuse source==output, existing output, existing sidecar
2. refuse an already-quantized source (`__metadata__` **and** inline `.comfy_quant` markers)
3. refuse a stale `.partial`
4. plan the output header, computing every `data_offsets` up front
5. stream `("write", tensor)` / `("copy", (start, size))`
6. assert bytes-written == bytes-planned
7. `flush` + `fsync` + `os.replace`, with `finally: partial.unlink()`
8. guard free RAM and free disk before starting

Plan/write/verify/replace lives at `quant_w4a4.py:196-231` + `:269-327`, `quant_w4a8.py:342-393`,
`quant_int8.py:246-289`, `quant_w4a4_smooth.py:263-312`, `quant_mixed.py:601-657`,
`to_native.py:185-227`. Helpers: `read_header` x6, `copy_range` x3 named + 2 inline, `read_tensor` x3
with **two different implementations**, `human_size` x3, `SAFETENSORS_DTYPE` x2 **differing**
(`quant_w4a8.py:211` has `int16`, `quant_mixed.py:65` does not).

## Where duplication has already produced divergence

- **The exclusion net exists in exactly one converter.** `quant_w4a4.py:37-46` defines `EXCLUSIONS`
  and applies it at `:161` and `:170`, with a docstring explaining why the allowlist alone is not
  enough. `quant_w4a8.py:147-161`, `quant_int8.py:75-87` and `quant_mixed.py:164-178` have none -- and
  `quant_int8.py:46-55` imports `PROFILE_PATTERNS` from `quant_w4a8`, so the loosest pattern table in
  the tree is shared by two converters and is the one with no second net. Latent today; live the first
  time anyone writes a looser pattern for Flux / SeedVR2 / Z-Image, all of which CLAUDE.md lists as
  pending.
- **The RAM guard exists in three of five**, and `quant_w4a4` uses a different formula (correct for
  its streaming design). `quant_w4a4_smooth.py` has none, despite `:200-247` accumulating `quantized`,
  `scales` and `new_norms` across every layer before writing -- the exact two-pass shape
  `_ram_guard.py` was written for.
- **The disk guard is three different rules.** `quant_w4a4.py:402-405` checks `estimate + 1 GiB` and
  reports what was free. `quant_w4a8.py:294`, `quant_int8.py:216`, `quant_mixed.py:543` check
  `< source size` and raise a bare `SystemExit` with no numbers. `quant_w4a4_smooth.py` and
  `svdq_to_bf16.py` do not check.
- **The dtype-writing fix landed in one copy.** `quant_w4a8.py:177-208` grew `header_dtype()` and
  `as_bytes()` for ticket 24, with an EXECUTED note at `:194-201` (bfloat16/fp8 `.numpy()` raises).
  `quant_mixed.py` calls the same `ck.quantize_w4a8_int8_weight` at `:565-569`, receives the same
  five-tuple, and writes it at `:618-620` + `:645` handling only `float8_e4m3fn`. A `float8_e5m2`
  scale `KeyError`s; a bfloat16 codebook raises inside `.numpy()` -- **after every layer has been
  quantized.**

## The proposed shape

```python
class Conversion:
    def __init__(self, source, output, *, preflight, sizer): ...
    def plan(self, header, metadata, per_layer): ...
    def commit(self): ...
```

`per_layer(name, tensor) -> list[(key, tensor)]` is the only plugged-in part: two lines for W4A4, six
for W4A8, nine for mixed.

`quant_int8.py:46-55` is the existence proof that the seam works -- it already imports six helpers
from `quant_w4a8`. The move is to finish what it started and invert the direction: the shared code
should not live inside whichever converter was written first.

## Deletion test

Delete `output_header` / `write_streamed_checkpoint` from `quant_w4a4.py` and complexity
**concentrates** -- callers become `plan()` + `commit()`. Delete `_ram_guard.py` and it
**redistributes** into three inline formulas that will drift again, which is what its own docstring
records. The write loops fail the test; `_ram_guard` passes.

## Cheapest immediate fix, if the owner wants to defer the core

`from quant_w4a8 import as_bytes, header_dtype` in `quant_mixed.py`, used at `:620` and `:645`. Three
lines. It does not solve the ticket -- it removes the one divergence that crashes after a full
quantization pass.

## Closing criterion (written before the work)

This ticket closes on a **decision**, not on code: the owner says core / no core / defer, and the
reason is written here. If "core", it spawns implementation tickets and does not itself close until
`grep -c "def write_streamed_checkpoint" tools/*.py` returns 1 and every converter's guard set is the
same set.
