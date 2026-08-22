# ComfyLite -- a folder beside ComfyUI, not inside it

## Destination

A decision-complete plan for ComfyLite phase 0: where it lives, how it sees the models, what it reuses
from this bench, and what it must not touch -- so that the first line of code is written against
answers rather than assumptions. **Not** the application itself.

## Notes

- **Source of truth for the design**: `ComfyLite_Project_Handoff.md` at the repo root, tracked as of
  `e37b1a1`.
- **Two decisions already made by the owner, 2026-08-22**: ComfyLite lives at
  `F:/COMFY_PORTABLE/ComfyLite/` **with its own git repo** (beside `ComfyUI/`, never inside it), and the
  stack is **Tauri + Svelte** as the handoff proposes.
- **Domain**: read `CLAUDE.md` and `AGENTS.md` first. ComfyLite is a separate repo and gets its own
  `CONTEXT.md` when it has vocabulary of its own -- do not add its terms to the bench glossary
  (`docs/agents/domain.md`).
- **The constraint that shapes everything**: "somando, sem afetar nada". Every ticket here states how it
  avoids touching the bench.

## Why "beside, not inside" is the right call, and not just a preference

`ComfyUI/` is a **separate git checkout** (`v0.33.0-19-gc1739380`) that the root repo does not track.
Anything placed inside it is untracked by both repositories and dies in a reinstall -- `AGENTS.md:39-43`
already says so. Outside it, three repos touch and none overlaps.

## Decisions so far

<!-- one line per closed ticket -->

## Not yet specified

- **The Retrieval Agent** (handoff section 8) and the **Source Adapters** (section 6). Nothing on this
  bench resembles either -- `fetch_ltx25.py` and `fetch_minimax_h3.py` are single-repo fetchers, not
  adapters, and there is no LLM-client code at all. `huggingface_hub 0.36.2` is installed and gives
  `HfApi.list_models` for free. Revisit after the inventory exists, because an adapter with nothing to
  compare against cannot decide "ja tenho essa variante".
- **The MCP server** (section 15). Depends on the shape of everything else.
- **LoRA intelligence** (section 12). 30 LoRA files over 100 MB across the two roots, no metadata
  anywhere. Blocked on the inventory.
- **The attention-autotuner database schema** (section 12). Three tools already cover three of its four
  axes -- `_check_accel.py` validates numerically without timing, `attn_bench.py` times without
  validating, `attn_dtype_ab.py` does both plus a dtype axis. Nothing persists a winner and nothing
  selects one. The missing piece is persistence + selection, and its schema waits on ticket 04's device
  stamping.
- **Whether ComfyLite ever needs `--disable-dynamic-vram` off.** Two independent workflows on this bench
  require it on; nothing has tested whether a ComfyLite-managed process could avoid it.

## Out of scope

- Building ComfyLite. This map ends when phase 0 is decision-complete.
- Reimplementing ComfyUI, or converting custom nodes to optimized Python. The handoff's own section 23
  rules both out and it is right.
- Changing anything under `ComfyUI/` or in the bench's tracked tree. ComfyLite reads; it does not write
  there.

## Tickets

- [01-seam-de-descoberta-de-modelos.md](issues/01-seam-de-descoberta-de-modelos.md)
- [02-scaffold-e-colisao.md](issues/02-scaffold-e-colisao.md)
- [03-inventario-duas-raizes.md](issues/03-inventario-duas-raizes.md)
- [04-participar-do-lock-da-gpu.md](issues/04-participar-do-lock-da-gpu.md)
- [05-runtime-compat-primeiro.md](issues/05-runtime-compat-primeiro.md)
