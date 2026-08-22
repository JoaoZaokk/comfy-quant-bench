# Repository Guidelines

## Project Structure & Module Organization

This directory is a live ComfyUI Portable installation, not a clean source-only project. `ComfyUI/` contains the Git checkout and core Python packages; follow its nested `AGENTS.md` when editing there. Models live under `ComfyUI/models/<category>/`, custom extensions under `ComfyUI/custom_nodes/`, workflows and user state under `ComfyUI/user/`, and tests under `ComfyUI/tests/` and `ComfyUI/tests-unit/`. Root `.bat` files launch the portable environment. Put reusable quantization utilities in `tools/`; keep generated model files in the same model category as their source.

## Build, Test, and Development Commands

Never use global Python, pip, or Conda. Run everything with the embedded interpreter:

- `.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build` starts the standard UI.
- `.\run_nvidia_gpu_8190_loopback.bat` starts the local Ampere-oriented instance on `127.0.0.1:8190`.
- `.\python_embeded\python.exe -m pytest ComfyUI\tests-unit` runs unit tests.
- `.\python_embeded\python.exe -m pytest ComfyUI\tests -m "not inference"` runs non-inference integration tests.
- `.\python_embeded\python.exe -m ruff check ComfyUI` applies the configured Ruff checks when Ruff is installed.

Do not mass-upgrade Torch, CUDA, ComfyUI, or related dependencies. Inspect versions and local APIs before proposing any package change.

## Coding Style & Naming Conventions

Use four-space Python indentation and match nearby code. Prefer small, direct changes, module-level imports, clear `snake_case` functions, and `PascalCase` classes. Avoid speculative abstractions and new dependencies. Core ComfyUI intentionally ignores Ruff line-length checks; prioritize readable local style.

## Quantization Safety & Tracking

Inspect before modifying. Never delete, move, overwrite, or requantize original models. W4A4 means actual ConvRot W4A4 execution through installed ComfyUI/comfy-kitchen APIs, not weight-only INT4 followed by BF16 compute. Quantize large compatible Linear tensors first; preserve sensitive tensors and architecture-specific exclusions. Maintain `W4A4_PROGRESS.md`, `quantization_inventory.json`, and a `.quant.json` sidecar for every output.

## Testing Guidelines

Name pytest files `test_*.py`. Validate each quantized model individually: file integrity and metadata, tensor layout/dtypes, normal ComfyUI loader compatibility, then a matched-parameter smoke test or benchmark. Record unsupported or negative results rather than hiding them.

## Commit & Pull Request Guidelines

The root **is** a Git repository as of 2026-08-19. Its `.gitignore` is an allowlist (`/*` then
re-include), because a denylist that misses one entry tries to commit a 42 GiB safetensors; models,
`python_embeded`, venvs and `ComfyUI/` stay out. Verify with `git add -An --dry-run` before any
`git add`.

**Do not quote the tracked set from this file — list it.** This paragraph read "only `tools/`,
`custom_nodes/`, the root `.md` files and two root scripts", which stopped being true once `docs/`,
`.scratch/` (the local issue tracker) and `calib/` joined. Same treatment CLAUDE.md already gives
the tracked-file count, for the same reason: it drifts faster than anyone edits prose.

```bash
git ls-files | wc -l                                                              # how many
git ls-files | awk -F/ 'NF==1{print "root"} NF>1{print $1"/"}' | sort | uniq -c   # where
```

`ComfyUI/` is a **nested checkout with its own remote**, and its `.gitignore:8` excludes
`/custom_nodes/`. Git will not descend into it, so nothing under `ComfyUI/` can be tracked from
the root repository — anything written there is invisible to both and dies in a reinstall. That is
why `custom_nodes/comfy-quant-preflight/` lives at the root and a three-line stub inside
`ComfyUI/custom_nodes/` loads it. Commits to ComfyUI itself still belong to that checkout or to an
individual custom-node repository. Follow the existing short imperative subjects (`Fix ...`, `Add ...`, `Support ...`). PRs should state the problem, behavioral change, tests run, model/hardware affected, and before/after measurements; include screenshots only for visible UI or output changes.
