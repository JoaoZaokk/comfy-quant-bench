# 03 - The inventory scanner: two roots, header-only reads, and a hash ladder

Type: task
Status: ready-for-agent
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
