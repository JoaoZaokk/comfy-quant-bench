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
