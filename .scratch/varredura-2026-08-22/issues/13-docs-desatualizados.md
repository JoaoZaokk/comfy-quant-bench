# 13 - Nine documents carry numbers, versions or ticket counts that no longer match the tree

Type: task
Status: resolved
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

## Closed 2026-08-22, commit `13fbd8c` -- and the priority item was fixed the wrong way first

Counts recounted or replaced with the command that produces them, following the precedent CLAUDE.md
set for the tracked-file count. The two UPSTREAM_REPORT files carry the installed comfy-kitchen
version, read from site-packages.

### The `W4A4_HANDOFF.md:22` item, and why it is worth reading twice

It was the priority item, and the first fix **replaced one false statement with another**. The new
text read *"OBSOLETE. The file is gone, and it must not be put back"*, evidenced by
`find python_embeded -iname "cudart64*.dll"` returning only `cudart64_13.dll`.

**That pattern cannot match a name ending in `.disabled`.** The file is at
`torch/lib/cudart64_12.dll.disabled`, 556,544 bytes, sha256 `d954ca54...cf9dad`, byte-identical to
the copy in the unrelated Ultravox venv. It was **renamed, not deleted** -- which
`W4A4_PROGRESS.md:339-340` had already recorded, from a run, in this same repository.

Three things follow, and they are the content of this ticket rather than a footnote to it:

1. The evidence command was **blind by construction**, and nobody checked that it could see what
   it was being used to rule out. This is `arquivo-plausivel-nao-e-o-caminho` and CLAUDE.md's own
   rule, broken in the two documents that state it.
2. The correct fact was **already written down here**, executed, and the change overwrote it
   without reconciling the two.
3. The sha256 was deleted on the reasoning that a hash under a "do not restore" heading is an
   invitation. Backwards: restoring is a one-command rename either way, and the hash was the only
   thing identifying the unlabelled artifact still sitting in `torch/lib/`. Restored.

The runtime advice was right the whole time -- Windows will not load a `.dll.disabled`, and
`_check_accel.py` returns ALL GOOD on the 3090 with it disabled. It was the **record** that was
wrong, which is the harder kind to notice.
