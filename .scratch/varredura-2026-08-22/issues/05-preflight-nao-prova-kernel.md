# 05 - Six definitions of "is the native backend ready", and the flagship converter uses the weak one

Type: task
Status: ready-for-agent
Blocked by: -
Severity: medium
Provenance: TRACED, then EXECUTED 2026-08-22 on the 3090; one sub-claim needs a GPU run

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

## Answer, partial -- EXECUTED 2026-08-22 on the RTX 3090, GPU lock held

Two of the three questions in this ticket are now settled, and **one of my claims is refuted.**

### The groupsize claim was WRONG. Resolution is invariant.

I wrote: *"the constraint validation runs against a configuration the conversion does not use...
If the CUDA backend's constraint set is groupsize-sensitive, a conversion at 256 is preflighted at
64 and the answer means nothing."* Measured, all four combinations:

    config          quantize impl                  linear impl
    cg=64  real     comfy_kitchen.backends.cuda    comfy_kitchen.backends.cuda
    cg=64  dummy    comfy_kitchen.backends.cuda    comfy_kitchen.backends.cuda
    cg=256 real     comfy_kitchen.backends.cuda    comfy_kitchen.backends.cuda
    cg=256 dummy    comfy_kitchen.backends.cuda    comfy_kitchen.backends.cuda

One distinct implementation across all four. And the real calls succeed at both groupsizes --
`cg=64` and `cg=256` both produce finite bf16 output of the right shape.

**So `quant_w4a4.py`'s 64/64 preflight is untidy, not wrong.** The ticket's severity drops from
medium to low on this axis.

### The dummy-vs-real claim is ALSO not demonstrated -- and it is in `_native_probe.py`'s docstring

`_native_probe.py:9-15` asserts that dummy kwargs let the check pass while a real call would drop
to eager, citing `comfy_kitchen/registry.py:246` (empty/None kwargs skip constraint validation).
Measured here: **`torch.empty` and real quantized tensors resolve identically**, for both ops, at
both groupsizes.

State this carefully, because it is the exact shape of error this repo keeps making: I tested
**two ops, two groupsizes, one build, one card**. That is not "the claim is false" -- it is "the
claim did not reproduce under the only conditions anyone has tried." The mechanism in
`registry.py:246` may still be real for other ops or other constraint sets. **The docstring should
say the claim is a reading of `registry.py`, not a measurement**, which is what it currently
implies.

### `_native_probe.native_backend_ready()` was executed for the first time, and it works

Its own docstring says *"neither function below has been executed by it -- only `py_compile` and
import-without-CUDA-op were run."* It has now been run. It returns
`native_ready: true`, resolving both ops to `comfy_kitchen.backends.cuda`, and lists the backend's
op set. **The shared probe is not vapourware.** Update its docstring.

### What is still open, and is the real content of this ticket

Six definitions remain, and **no converter imports the shared one**. That is unchanged and is why
this ticket stays open. The argument for consolidating is now *simpler*, not weaker: since
resolution is invariant to the things the six copies disagree about, there is no defensible reason
for six of them.

### Closing criterion, revised

Items 1-3 stand. **Item 4 is answered: no difference at 64 vs 256.** Write that into
`_native_probe.py`'s docstring next to the code, with the date and the card, so the next session
does not re-derive it.
