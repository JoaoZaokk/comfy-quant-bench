# 01 - How does ComfyLite see the same models ComfyUI sees?

Type: grilling
Status: resolved
Blocked by: -
Provenance: EXECUTED for the import test; OBSERVED for the filesystem

## Question

Two options, and the answer fixes the coupling for the life of the project.

**Option 1 -- import `folder_paths`, then load the YAML.** This is what three tools on this bench
already do, and the reason is written in the code at `tools/diffusion_smoke.py:56-64`:

> `extra_model_paths.yaml` is read by `main.py` at boot, not by importing `folder_paths`, so a script
> that skips `main.py` sees only `ComfyUI/models`. Register it here or every model on the D: mount is
> invisible.

Same three lines at `tools/calibrate_activations.py:317-321` and `tools/dispatch_census.py:169`.

**Measured 2026-08-22, executed, not traced:**

```
torch in sys.modules: False
argv survived: ['-c']
new heavy modules: ['comfy', 'comfy.cli_args', 'comfy.options']
categories: 27
```

So `import folder_paths` pulls no torch, does not consume `sys.argv` (because
`comfy/options.py:2` initialises `args_parsing = False` and only `main.py:2` flips it), and yields
27 categories. It is cheap.

`sys.path` is already solved: `python_embeded/python313._pth` line 1 is `../ComfyUI`, so **any** script
run with the embedded interpreter has ComfyUI importable from any working directory, before the first
line executes.

**Option 2 -- re-parse the YAML, no ComfyUI import.** `ComfyUI/utils/extra_config.py:6-33` is 28 lines
and trivial to reimplement. The part that is not trivial is `ComfyUI/folder_paths.py:25-66`, the
default category table, which encodes: `text_encoders` searches **two** directories
(`text_encoders` and `clip`); `diffusion_models` searches **two** (`unet` and `diffusion_models`);
`controlnet` searches two (`controlnet`, `t2i_adapter`); `diffusers` uses the sentinel extension
`["folder"]`; `classifiers` uses `{""}`; `supported_pt_extensions` has 8 entries including `.pt2`,
`.pkl`, `.sft`.

## Recommendation

**Option 1.** A hand-copied table is stale the next time `update_comfyui.bat` runs, and it fails
**silently** -- a category quietly returns fewer files. That is the exact failure shape this bench
exists to avoid. Option 1's coupling is two symbols, `folder_paths.folder_names_and_paths` and
`utils.extra_config.load_extra_path_config`, both stable across the 0.29 -> 0.33 window this install
already crossed.

Shape: the Python worker imports them and exports the resolved map as JSON to the Rust side. Drift
becomes impossible by construction.

## Three traps that come with Option 1, all measured

1. **`unet` and `clip` are not separate categories.** `folder_paths.map_legacy` (`:112-115`) rewrites
   `unet` -> `diffusion_models` and `clip` -> `text_encoders`, and `add_model_folder_path` calls it
   first (`:375`). The YAML lists both, so a naive reader double-counts.
2. **Six YAML keys are not built-in categories** -- `ipadapter`, `pulid`, `insightface`, `inpaint`,
   `ultralytics`, `gguf`. `add_model_folder_path:388-389` creates them with an **empty extension set**,
   and `filter_files_extensions` (`:436-437`) passes *everything* when the set is empty. Those six list
   `.json`, `.txt`, `.md` as models.
3. **`import folder_paths` creates `ComfyUI/input` if missing** (`:117-121`). Today a no-op -- the
   directory exists. It stops being one if anyone deletes it.

And one hard rule for the scaffolding: **never import anything from `tools/` to do this.**
`_check_accel.py` has no `if __name__ == "__main__"` guard and runs a full attention battery at module
scope. Importing it takes the GPU.

## What "every model ComfyUI can see" actually means here

The two are already different, measured 2026-08-22:

- `ComfyUI/models` -- **623 GiB, 424 files**, 72 top-level directories, of which roughly 45 are
  custom-node-owned categories (`BiRefNet`, `sam2`, `sam3`, `matanyone`, `liveportrait`, `reactor`,
  ...) that `folder_paths` does not know and the YAML does not map. Custom nodes reach them by
  hardcoded path.
- `D:/ComfyUI-Models` -- **409 GiB, 81 files**, mapped by the YAML's single `viral_d` block. Three
  directories that exist there are **not** in the YAML and so are invisible to ComfyUI: `diffusers`
  (12 GB), `latent_upscale_models` (1.2 GB), `model_patches` (4 MB).

**Combined: ~1.03 TiB, 505 model files.** So "what ComfyUI can see" and "what is on disk" differ by
about 12 GB in one direction and by ~45 uncatalogued directories in the other.

**This is the decision, and it is the owner's:** does ComfyLite's inventory mean *ComfyUI's view*
(authoritative, category semantics, blind to 45 directories) or *the disk* (complete, no semantics)?
The honest answer is probably both, in one record, with a `visible_to_comfyui` flag -- but that is a
choice, not a default.

## Closing criterion

Closed when the owner picks Option 1 or 2 and picks the inventory's meaning, and both are written
here. No code required to close.

## RESOLVED, 2026-08-22 -- the owner picked Option 1

**Seam: import `folder_paths`.** The worker puts `comfy_root` on `sys.path`, imports
`folder_paths` and `utils.extra_config`, calls `load_extra_path_config` so the D: mount becomes
visible, and exports the resolved map as JSON to the Rust side. Coupling stays at the two symbols
named above. The reason the owner's answer matches the recommendation is the failure *mode*, not
the effort: a hand-copied category table does not break loudly on the next `update_comfyui.bat` --
a category quietly returns fewer files, and nothing prints.

**Inventory meaning: the union, in one record, with `visible_to_comfyui` per row.** Both views are
kept because they genuinely disagree in *both* directions -- ~45 custom-node-owned directories that
`folder_paths` does not know, and three directories on D: (`diffusers`, `latent_upscale_models`,
`model_patches`) that the YAML never maps. Recording either view alone throws the other away
permanently; recording the union throws nothing away and costs one integer column.

This half of the answer was **not** stated by the owner in the same breath as the seam. It is the
shape ticket 03's closing criterion already demanded before anyone looked at a result -- *"every
record carries the root it came from and a `visible_to_comfyui` flag"* -- so it is taken as the
written-in-advance criterion rather than as a fresh decision. **If the owner wants ComfyUI's view
alone, this is the line to contradict**, and the change is a filter at read time, not a re-scan.

Implemented in `ComfyLite/worker/comfylite/discovery.py` and `catalog.py` (the `visible_to_comfyui`
column). Unblocks tickets 02 and 03.
