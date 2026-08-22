# 15 - quant_audit covers one of the two model roots, so the checked-in inventory is missing 409 GiB

Type: task
Status: resolved
Blocked by: -
Severity: medium
Provenance: OBSERVED 2026-08-22

## Problem

`tools/quant_audit.py:40` defaults `--models-root` to `ComfyUI/models`, and `:478-479` does
`path.relative_to(models_root)` -- so it is structurally single-root, not merely configured that way.

The bench has **two** roots. `ComfyUI/extra_model_paths.yaml` maps `D:/ComfyUI-Models/`, which exists
and holds **409 GiB across 81 files** (34 of them `.safetensors`, one `.gguf`), including a 326 GiB
`diffusion_models` and a 54 GiB `text_encoders`.

Measured: `grep -c "ComfyUI-Models" quantization_inventory.json` returns **0**.

So the inventory that CLAUDE.md instructs every session to regenerate and trust describes 623 GiB of a
1.03 TiB installation, and nothing in its output says so.

Two smaller gaps in the same tool, both from the same single-root shape:

- **No hashes.** Grep for `hashlib|sha256|md5` over the file: 0 hits. `tools/model_audit.py:111-122`
  already has the right ladder (size -> `blake2b` of first+last 1 MiB -> full `sha256` only on
  collision) and is a separate tool.
- **72 top-level directories under `ComfyUI/models`**, of which roughly 45 are custom-node-owned
  categories (`BiRefNet`, `sam2`, `sam3`, `matanyone`, `liveportrait`, `reactor`, ...) that
  `folder_paths` does not know and the YAML does not map. An inventory built on
  `folder_paths.get_filename_list()` is blind to them; one built on a disk walk sees them with no
  category semantics.

## Why this is `medium` and not `low`

It is the one finding where the tool is *silently incomplete* rather than wrong. Everything downstream
that asks "do we already have a compatible variant" -- which is the whole of ComfyLite phase 1 -- gets
a "no" that means "not in the half I looked at".

## Closing criterion (written before the fix)

Closed when:

1. `quant_audit.py` accepts multiple roots, and by default reads them from
   `ComfyUI/extra_model_paths.yaml` rather than requiring them on the command line;
2. every record carries which root it came from, so `relative_to` is unambiguous;
3. the output states the roots it covered and the count per root, on every run;
4. `grep -c "ComfyUI-Models" quantization_inventory.json` is greater than zero after a regeneration;
5. the tool states in its output that it did not hash, or it hashes using `model_audit.py`'s ladder.

Item 3 alone is what stops this recurring: an inventory that names its own scope cannot be misread as
complete.

## Closed 2026-08-22, commit `13fbd8c` -- with one refusal restored

Multi-root, roots read from `extra_model_paths.yaml` by default, every record carrying its root,
and the output stating the roots it covered with a count per root. Item 3 -- an inventory that
names its own scope -- is the one that stops this recurring, and it is done.

### The regression: a hard refusal became a warning

Going multi-root replaced `if not models_root.is_dir(): raise SystemExit(...)` with a check that
fired only when **every** root was missing. So a typo'd `--models-root` landed in
`roots_declared_but_missing` and the run continued -- overwriting `quantization_inventory.json`,
the default output and a tracked artifact, with a partial walk, at exit 0. On a default run with
the D: mount offline that regenerates exactly the 623-of-1030-GiB single-root inventory this
ticket exists to eliminate.

Restored, and the two cases separated, because they are not the same shape:

- **a path the operator named on the command line and that does not exist** -> refuse. They asked
  for it by name; continuing silently substitutes a different question.
- **a root declared in the YAML that is not there** -> warn, and record it in stdout, the JSON and
  the markdown. The D: mount may simply be offline.

Verified: `--models-root <typo>` alongside a good root now exits 1 with the path named, and writes
no JSON.

### Still open

Criterion item 4 -- `grep -c "ComfyUI-Models" quantization_inventory.json` above zero -- needs a
real regeneration over 1.03 TiB, which was not run. The tool was exercised on a synthetic tree.

## Item 4 closed by execution, 2026-08-22 (round 2), commit `4480337`

The one item this ticket left open needed a real regeneration over ~1 TiB, and it ran, on the 3090
with the lock held (`stack_snapshot()` allocates on CUDA, so the inventory cannot be rebuilt while
a sibling holds the card -- worth knowing before planning one):

    Covered 2 root(s), 199 files, 1.01 TiB:
      ComfyUI/models   164 files   622.11 GiB   F:\COMFY_PORTABLE\ComfyUI\models
      viral_d           35 files   408.29 GiB   \\192.168.3.68\estoque\ComfyUI-Models

`grep -c "ComfyUI-Models" quantization_inventory.json` -> **72**, not 0. `schema_version: 3`.
And the footer states what it did not do: no content hash was computed, headers only, so
identical-looking entries are not known to be identical files.

### The regeneration surfaced something nobody had written down

**`D:` is not a disk.** `net use` reports `D: -> \\192.168.3.68\estoque`. The audit resolved the
declared `D:\ComfyUI-Models` to that UNC and recorded **both** the declared and the resolved path,
which is the right shape and is how it became visible at all.

So 408 GiB of the model set arrives over SMB, from a NAS that is not the one holding the ERP code
(`192.168.3.40`). Two consequences: a full walk costs network, and **"the D: mount is offline" is a
normal state rather than a broken one** -- which is exactly why this ticket's own regression fix
distinguishes a declared-and-absent root (warn) from a root the operator named on the command line
(refuse). CLAUDE.md corrected.
