# 13 - Nine documents carry numbers, versions or ticket counts that no longer match the tree

Type: task
Status: ready-for-agent
Blocked by: 02
Severity: low
Provenance: TRACED

## Problem

Separate from ticket 02, which is about a doc claim that changes what a reader *does*. These are
numbers that have simply drifted. They matter here more than in a normal repo because every session
reads them as input, and this repo has a recorded history of exactly this failure -- a tracked-file
count wrong three times in one session, a `custom_nodes` count wrong three times, a hard rule whose
stated reason was wrong for five days.

- `W4A4_HANDOFF.md:22` still instructs a reader to install a CUDA 12.6 DLL that CLAUDE.md now records
  as obsolete and already deleted, with the evidence that all four accel extensions import without it.
  **This is the one to fix first** -- it tells someone to put a file back.
- `UPSTREAM_REPORT_w4a8_capture.md:6` and `UPSTREAM_REPORT_dtype_widget.md` carry a comfy-kitchen
  version eight releases behind the installed 0.2.31. Both are unpublished; publishing them with a
  stale version invites a "cannot reproduce".
- `INVENTARIO_BANCADA.md:86` carries `custom_nodes` counts that CLAUDE.md deliberately stopped quoting.
  The number moved out of one file and stayed in another.
- `.scratch/estado-entregavel/triagem-89.md:174` claims every tracked file has a classification line;
  35 tracked files do not appear.
- `.scratch/estado-entregavel/artefatos-em-comfyui.md:127` attributes the tracked package's byte size
  to the ComfyUI side of the split.
- `AGENTS.md:33` and `W4A4_HANDOFF.md` describe the tracked set in a way that predates `docs/` and
  `.scratch/` joining it.
- `CORTIQ_LTX25_HANDOFF.md:8` points at a 17-ticket plan; the map has 29.
- Two of nine upstream line numbers cited across docs and `checks.py` have drifted (seven are exact).
- `.scratch/estado-entregavel/issues/16-...md` -- `comfy_convrot_native/` is still orphaned from both
  repositories and the ticket is marked resolved.

## Closing criterion (written before the fix)

Closed when every number above is either **recounted at edit time** or **replaced with the command
that produces it**, following the precedent CLAUDE.md already set for the tracked-file count and the
`custom_nodes` count ("count it, do not quote it"). A number that will drift again should not be
written down a second time.

Specifically: `W4A4_HANDOFF.md:22` no longer tells anyone to restore `cudart64_12.dll`.
