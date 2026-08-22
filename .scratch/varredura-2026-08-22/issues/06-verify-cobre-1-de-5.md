# 06 - verify_w4a4 smoke-tests a rotation the file may not use, and covers one of five output formats

Type: task
Status: ready-for-agent
Blocked by: 05
Severity: medium
Provenance: TRACED

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
