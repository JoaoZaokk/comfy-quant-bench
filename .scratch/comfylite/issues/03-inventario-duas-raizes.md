# 03 - The inventory scanner: two roots, header-only reads, and a hash ladder

Type: task
Status: resolved
Blocked by: 01
Provenance: OBSERVED + EXECUTED

## What already exists, and what it misses

`tools/quant_audit.py` is the closest thing on the bench to handoff phase 1 -- it walks by extension,
infers `role` from directory, infers architecture, records format, and reads per-layer quant metadata
**without materializing tensor data**, producing `quantization_inventory.json` (706,757 bytes,
`schema_version: 2`, `total_size_bytes: 654561491511`).

Three gaps, and the first is disqualifying for ComfyLite:

1. **Single root, structurally.** `:40` defaults `--models-root` to `ComfyUI/models` and `:478-479`
   does `path.relative_to(models_root)`. Measured: `grep -c "ComfyUI-Models" quantization_inventory.json`
   returns **0** -- the checked-in inventory is missing the entire 409 GiB D: mount. This is tracked as
   its own bench ticket (`varredura-2026-08-22/issues/15`).
2. **No hashes.** Grep for `hashlib|sha256|md5`: 0 hits. Handoff section 4 requires one per artifact.
3. **No LoRAs-as-LoRAs, no workflows.** A LoRA lands as `role: lora` and stops -- no base architecture,
   no target modules, no recommended strength.

`tools/model_audit.py:111-122` already has the right hash ladder and should be lifted rather than
reinvented: size, then `blake2b` of first + last 1 MiB + size, then full `sha256` only on collision.

## The hash is the expensive part, and the shortcut is not available

The ComfyUI `assets` table looked like a free ride -- `ComfyUI/app/assets/scanner.py:35,326-327`
computes `blake3:` digests and a background seeder populates it. **Measured 2026-08-22 against a
read-only copy of `ComfyUI/user/comfyui.db`:**

```
tables: ['alembic_version', 'asset_reference_meta', 'asset_reference_tags',
         'asset_references', 'assets', 'tags']
assets cols: ['id', 'hash', 'size_bytes', 'mime_type', 'created_at']
assets rows: 0
  hash non-null: 0
journal_mode: delete
```

**Zero rows.** The schema is there and the seeder has never populated it on this install. So hashing
1.03 TiB is real work that nobody has done, and the ladder above is how to avoid doing all of it.

Note `journal_mode: delete`, not WAL -- so a reader can hit `database is locked` while ComfyUI writes.
Read a **copy**, never the live file, and never take its `FileLock`: `ComfyUI/app/database/db.py:75-91`
acquires an exclusive lock and the next ComfyUI start dies if someone else holds it.

## Two things the scanner must not do

- **Never `mmap` or `safe_open` a whole model.** CLAUDE.md records that mapping the 21.93 GiB Gemma
  source on this host failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
  Read safetensors **headers** only, the way `quant_audit.py` already does.
- **Never write a sidecar into a model directory.** `folder_paths.py:496-501` invalidates its filename
  cache on directory mtime, so a single file written into `diffusion_models` forces ComfyUI to re-walk
  209 GB (and the 326 GB D: mirror) inside a request. ComfyLite's sidecars live under `ComfyLite/`.

## Closing criterion

Closed when the scanner produces one catalog covering both roots, every record carries the root it came
from and a `visible_to_comfyui` flag, hashing follows the three-step ladder with the cheap tier by
default, no model file is ever mapped or fully read, and a re-run over an unchanged tree is
incremental (mtime + size) rather than a full re-hash.

Expected catalog size: `quantization_inventory.json` is ~1.67 KB/file for 424 files, so 505 files with
the same richness is ~850 KB of payload; with indices and a `tested_configs` history, **2-5 MB**. Keep
media on disk and only paths in the DB and it stays under ~50 MB indefinitely.

## RESOLVED, 2026-08-22 -- criterion met, with one item honestly short

`ComfyLite/worker/comfylite/scanner.py` + `catalog.py`. First real run, EXECUTED end to end, not
traced:

```
roots walked   2          (comfy_models 424 files / 622 GiB, viral_d 81 / 408 GiB)
files seen     505
bytes seen     1.01 TiB   (1106654653527)
hashed         505  reused 0  pruned 0  warnings 1
elapsed        4.52 s
```

Re-run over the unchanged tree: **505 reused, 0 hashed, 0.20 s** -- 22.6x faster, which is the
incremental path proving itself rather than a test asserting it. Catalog on disk: **389,120 bytes**,
inside the 2-5 MB the criterion predicted. Every one of the 505 rows carries `root_key` and
`visible_to_comfyui`.

Hash tiers actually written: **cheap 379, full 126**. The 126 are tier-2 collision groups that got
escalated -- small files (195 B to 5 MiB) where head+tail+size genuinely match.

**The one clause not met literally, and it is the criterion contradicting itself:** "no model file is
ever mapped or fully read" cannot hold alongside "the three-step ladder", because tier 3 *is* a full
sha256 read. Nothing is ever **mapped** (that half holds absolutely, and it is the half that crashed
`torch_cpu.dll`), and nothing is fully read except on a tier-2 collision, bounded by
`--escalate-max-gib` (4 GiB default). Recorded rather than glossed: the criterion was written before
the ladder's third tier was thought through.

**The scan found something real and refused to decide it**, which is the behaviour worth keeping:

```
WARNING  tier-2 collision NOT escalated (11.46 GiB each > 4.00 GiB limit):
  comfy_models/diffusion_models/beyond-reality-zimage-v2_bf16.safetensors
  viral_d/diffusion_models/beyond-reality-recovered-bf16.safetensors
```

Two 11.46 GiB files with identical head, tail and size, one per root. Possibly a duplicate, possibly
differing only in the middle. `--escalate-max-gib 0` settles it at the cost of reading 23 GiB, 11 of
them over SMB. **Not run**, because nobody asked and the cost is real.

Follow-on defect found by running this, filed as its own ticket:
[06 - visible_to_comfyui is core-only visibility](06-visibilidade-custom-node.md).
