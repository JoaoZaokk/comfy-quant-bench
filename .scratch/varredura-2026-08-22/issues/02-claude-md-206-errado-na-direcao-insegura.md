# 02 - CLAUDE.md's preflight paragraph asserts three defects the code already fixed

Type: task
Status: ready-for-agent
Blocked by: -
Severity: medium
Provenance: EXECUTED 2026-08-22 (read the current files)

## Problem

`CLAUDE.md` (the *How the converter is built* section) states:

> Two more converters run **no** preflight at all -- `quant_w4a4_smooth.py` (which writes the same
> `convrot_w4a4` format) and `quant_int8.py` [...] and note that `quant_w4a4_smooth.py` does not
> write one [a `backend` field].

All three claims are false against the current tree:

- `tools/quant_w4a4_smooth.py:182-191` imports `normal_comfy_backend` from `quant_w4a4` and
  hard-refuses on `not native_ready`. Its comment at `:175-181` explains why, and names the audit
  that found "8 diverging copies of this probe with 3 different definitions of ready".
- `tools/quant_w4a4_smooth.py:328-330` writes `"backend"` and `"backend_linear"` into the sidecar,
  with a comment saying it was added precisely because its outputs were the only `convrot_w4a4`
  checkpoints with no record of which backend produced them.
- `tools/quant_int8.py:187-198` preflights when `--device cuda --convrot`, and prints why the
  `--no-convrot` path is exempt instead of pretending to preflight it.

The line references in the same paragraph have also drifted: `verify_w4a4.py:64` is now `:70`, and
`quant_w4a4.py:93-94` is now `:98-99`.

`git log -- tools/quant_w4a4_smooth.py tools/quant_int8.py` shows the change landed in `d14ae48`
("Close round 3, and run on the card what the agents could not"). The document did not follow.

## Why it is worth a ticket rather than a one-line edit

The drift is in the **unsafe direction**, and the consequence is concrete. A reader acting on this
paragraph would either add a duplicate preflight to a file that already has one -- making a seventh
copy of the problem ticket 05 is about -- or would distrust a sidecar `backend` field that is in fact
present, and re-run a conversion that did not need re-running.

Two other CLAUDE.md claims fail the same way and belong in the same edit:

- **CLAUDE.md says `tools/verify_w4a4.py` ends its output with what it did not cover, on every run.**
  It does not. `verify_w4a4.py:263-275` is a column of PASS lines with no caveat block. Grep for
  `not cover|NOT covered|Not checked|caveat` over `tools/` hits `comfy_run_workflow.py:539`,
  `ltx_studio.py:458`, `test_comfy_run_workflow.py:452` and `svdq_to_bf16.py:69` -- and nothing in
  `verify_w4a4.py`. This is the *opposite* direction from the paragraph above: the doc claims a
  safety practice the tool does not have.
- **The two audit-derived preflight checks that must stay WARN with "not confirmed by execution".**
  One of the two no longer carries that wording (`custom_nodes/comfy-quant-preflight/checks.py`).

## Closing criterion (written before the fix)

Closed when a single commit does all three, and none before the others:

1. the CLAUDE.md paragraph states what the three converters actually do today, with line numbers
   re-read at the time of the edit, not copied;
2. either `verify_w4a4.py` grows the caveat block CLAUDE.md promises, **or** the CLAUDE.md sentence is
   corrected to say which tools actually print it -- and the choice is stated in the commit body;
3. the preflight WARN text is re-read and either restored or the doc corrected.

Do not close by editing the doc alone if the fix is to make the tool honest -- pick one and say which.

## Answer to the second claim -- EXECUTED 2026-08-22, RTX 3090

**CLAUDE.md's "`verify_w4a4.py` ends its output with what it did not cover, on every run" is false,
and now measured rather than grepped.**

Full stdout of a real run against `zimage-v2-w4a4.safetensors` (170 layers, cg=256), with
`--kernel-smoke`, GPU lock held:

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

That is the whole output. Two PASS lines and a JSON object. Nothing states what was not covered --
and what was not covered is substantial: one layer of 170 was smoked, the byte-identity check
covers only preserved tensors, and (per ticket 06) the whole tool refuses any format but
`convrot_w4a4`.

This is the exact shape CLAUDE.md itself warns about: *"a column of PASS lines otherwise reads as
verified."*

### And a third CLAUDE.md claim, closed in the other direction

CLAUDE.md's environment section says of the removed `cudart64_12.dll`: *"an import proves DLL
resolution, **not kernel execution** -- `_check_accel.py` does the forward-and-compare and needs
the card, and **was not run**."*

**It has now been run**, on the 3090, with `CUDA_VISIBLE_DEVICES=0`:

    [OK  ] triton         v3.7.1 kernel compiled+ran
    [OK  ] sageattention  mean|d|=0.0006 vs SDPA (INT8 approx)
    [OK  ] flash_attn     mean|d|=0.0000 (causal 0.0000) vs SDPA
    [SKIP] xformers       not installed (bonus)
    RESULT: ALL GOOD

So the accel stack executes kernels without the CUDA 12.6 DLL. That caveat can be replaced with
the measurement and its date. **Ticket 13's "fix `W4A4_HANDOFF.md:22` first" becomes unambiguous:
that document tells a reader to restore a DLL that is now measurably unnecessary.**

### Closing criterion, revised

Item 2's fork is now decidable with evidence: `verify_w4a4.py` does **not** print the caveat, so
either add it or correct the sentence -- but the sentence as written is false and cannot stay.
