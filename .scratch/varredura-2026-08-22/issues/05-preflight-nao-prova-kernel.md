# 05 - Six definitions of "is the native backend ready", and the flagship converter uses the weak one

Type: task
Status: ready-for-agent
Blocked by: -
Severity: medium
Provenance: TRACED; one sub-claim needs a GPU run

## Problem

`normal_comfy_backend` exists six times, and the six do not answer the same question:

| file:line | ops resolved | probe tensors | groupsize probed |
|---|---|---|---|
| `quant_w4a4.py:89-109` | quantize + linear | `torch.empty` | **64 / 64** |
| `quant_w4a8.py:97-114` | quantize only | `torch.empty` | 16 / 256 |
| `quant_int8.py:102-136` | int8 convrot quantize | `torch.empty` | passed through (correct) |
| `quant_mixed.py:85-133` | all four | **real tensors** | 256 / 64 |
| `verify_w4a4.py:62-70` | **linear only** | `torch.empty` | 64 / 64 |
| `_native_probe.py:54-98` | quantize + linear | **real tensors** | 256 / 64 |

`_native_probe.py:9-15` names this problem in its own docstring and says its version follows
`quant_mixed`'s pattern rather than `quant_w4a4`'s, because `registry.get_implementation`'s `kwargs`
exist for *constraint validation* -- and `comfy_kitchen/registry.py:246` documents that empty or None
kwargs **skip** that validation. Dummy kwargs make the check pass while the real call would not.

**And then no converter imports it.** `_native_probe` is imported only by `convrot_ops_probe.py:22`,
`diffusion_smoke.py:23`, `gemma_chat.py:30`, `stage_probe.py:20`, `te_smoke.py:22` -- the probes.
The flagship converter still runs the version `_native_probe`'s own docstring calls the weak one.

## The concrete consequence

`quant_w4a4.py:66-76` exposes `--convrot-groupsize`, default 256, threaded into the real call at
`:302`. Its preflight at `:98-99` hardcodes `convrot_groupsize: 64, quant_group_size: 64`. **So the
constraint validation runs against a configuration the conversion does not use.**
`quant_w4a4_smooth.py:184` inherits it exactly.

`needs_run: true` on whether the CUDA backend's constraints actually differ at 64 vs 256 -- that is
not readable from the source. Settled by resolving both configurations and comparing
`impl.__module__`. Needs the card, so it needs a GPU window.

## Closing criterion (written before the fix)

Closed when:

1. exactly one `normal_comfy_backend` remains in the tree -- `_native_probe.native_backend_ready`,
   generalized to take the op names and **the real kwargs the caller is about to use**;
2. every converter and `verify_w4a4.py` call it with its own `args.convrot_groupsize` /
   `args.group_size`, so no tool can preflight a configuration it is not about to run;
3. `grep -c "def normal_comfy_backend" tools/*.py` returns 1;
4. the 64-vs-256 constraint question is answered by a run, in a GPU window, and the answer is written
   into `_native_probe.py`'s docstring next to the code -- not into the chat.

Item 4 may come back "no difference". That still closes the ticket: the fix is right either way, and
the answer stops the next session re-deriving it.

## Related

`K10`: `convrot_ops_probe.py:141` exits 0 if *any one* of five dispatch cases went native.
`CI-01`: `_native_probe.instrument()` (`:169`) silently no-ops its INT8 branch if a private
comfy_kitchen table moves. `CACHE-13`: `instrument` has no undo and caps distinct impls at 4.
