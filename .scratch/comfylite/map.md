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

- [01 - How does ComfyLite see the same models ComfyUI sees?](issues/01-seam-de-descoberta-de-modelos.md):
  **import `folder_paths`** (Option 1), and the inventory means **the union of both views with a
  `visible_to_comfyui` flag per row**, not ComfyUI's view alone. Owner picked the seam on 2026-08-22;
  the union half comes from ticket 03's pre-written closing criterion and is flagged in the ticket as
  the line to contradict if that was not intended.

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

- [01-seam-de-descoberta-de-modelos.md](issues/01-seam-de-descoberta-de-modelos.md) — **resolved**
- [02-scaffold-e-colisao.md](issues/02-scaffold-e-colisao.md) — 6 of 8 clauses met; open on the first
  commit and on opening a Tauri window (both the owner's call)
- [03-inventario-duas-raizes.md](issues/03-inventario-duas-raizes.md) — **resolved**
- [04-participar-do-lock-da-gpu.md](issues/04-participar-do-lock-da-gpu.md)
- [05-runtime-compat-primeiro.md](issues/05-runtime-compat-primeiro.md)
- [06-visibilidade-custom-node.md](issues/06-visibilidade-custom-node.md) — found by running 03:
  `visible_to_comfyui` is core-only visibility and under-reports by 110 GiB
- [07-subgraphs-bloqueiam-27-de-46.md](issues/07-subgraphs-bloqueiam-27-de-46.md) — found by running
  a REAL generation: subgraph expansion is unimplemented and blocks 27 of the 46 workflows

## Phase 3 is built and a real generation has run, 2026-08-23

The gap from phase 1 to phase 3 was closed in one pass: `comfyui.py` (client, UI->API conversion,
`/ws` relay), `workflows.py` (the 46 workflows, catalogued, schema v2), `loras.py` (34 LoRAs analysed
from headers), the generate endpoints and output proxy, and the Workflows + Generate screens.

**EXECUTED end to end on the 3090, with a real ComfyUI 0.33:** `txt_to_image_to_video` (SDXL + SVD),
`done` in 224.1 s, live progress relayed the whole way, **2 output files** fetched back through
ComfyLite's own proxy — a 1.37 MB PNG and a 469 KB MP4, both verified by magic bytes.

Two real bugs were found by that run and fixed; both are written up in ticket 07, and both are the
same shape: the failure cost GPU time before it became visible.

## The destination moved, and it should be said plainly

This map's destination was *"a decision-complete plan for phase 0 — **not** the application"*. On
2026-08-22 the owner asked for a beta, so phases 0, 1 and most of 2 were **built**, not planned:
~8,000 lines of Python worker plus ~4,000 of Svelte/Tauri, 125 tests green, and a real 505-file /
1.01 TiB catalog produced in 4.52 s. That is past this map's stated end.

Tickets 04 (GPU lock participation) and 05 (Compatibility Runtime) are untouched and are the next
real decisions — ComfyLite does not yet take the GPU or run a workflow, so neither has bitten.
