# Draft report: `w4a8_int8_linear` refuses CUDA graph capture above ~21.8M activation elements

**Status: written, NOT published.** Same rule as the other draft — no remote is configured here,
and filing this is an action directed outside the machine. It waits for the user.

comfy-kitchen 0.2.23, ComfyUI `v0.33.0-19-gc1739380`, RTX 3090 (sm86), torch 2.13.0+cu130,
Windows 11.

---

## Summary

`w4a8_int8_linear` captures into a CUDA graph for small activations and fails for large ones, with
`cudaErrorStreamCaptureInvalidated`. The boundary is on `M x K` — the activation element count —
and sits between 21.75M and 21.81M. `convrot_w4a4_linear` and a plain BF16 `F.linear` capture at
every size tested, including sizes above that boundary, so this is specific to the W4A8 tier.

The failure is inside `w4a8_codebook_linear_chunked`. A shape that skips that kernel and takes the
eager tail captures at exactly the size where a shape that uses it fails.

## Reproduction

```
python -s tools/graph_capture_probe.py --paths w4a8 --batches 5680 --shape 3840,3840
```

One M per process: a failed capture poisons the CUDA context, so every later call in the same
process raises the same cascaded error and later rows are artefacts rather than measurements.

## The boundary is on M x K

Bisected, one process per point:

| K | last M that captures | first M that fails | M x K |
|---|---|---|---|
| 3840 | 5664 | 5680 | 21,749,760 → 21,811,200 |
| 2560 | 8400 | 8600 | 21,504,000 → 22,016,000 |
| 1024 | 20800 | 21600 | 21,299,200 → 22,118,400 |

Three values of K, one of them not proportional to the other two, and the threshold lands in the
same band each time. `N` is 3840 throughout and does not enter.

## It is the chunked kernel, not the wrapper

`w4a8_int8_linear` drops to `eager_w4a8_int8_linear` when `out_features` is not divisible by 8 —
measured separately. That gives a control: same op, same size, different internal path.

| shape | internal path | capture at M=5856 |
|---|---|---|
| N=3840, K=3840 | `w4a8_codebook_linear_chunked` | **fails** |
| N=3841, K=3840 | eager tail | **captures** |

Both were confirmed by wrapping the `_C` entry points and reading their return values, not by
inferring from timing.

## Things ruled out

* **Not contamination.** A fresh process running only this op at only the failing M fails the same
  way.
* **Not alignment of M.** M=4097 captures; M=5855 and M=5857 both fail.
* **Not an algorithm switch visible from outside.** Eager time is flat across the boundary — 1182
  µs at M=5600, 1192 at M=5664, 1154 at M=5680 (the first failing one) — all inside each other's
  spread over three timed passes. And the Python-level dispatch is identical on both sides: only
  `w4a8_codebook_linear_chunked` is called, returning `True`.

## Not determined

The mechanism. `cudaErrorStreamCaptureInvalidated` reports that something invalidated the capture,
not what; the original error is swallowed by the cascade, and `CUDA_LAUNCH_BLOCKING=1` does not
surface it. Typical causes are a synchronizing call, a launch on a non-captured stream, or a raw
`cudaMalloc` inside the capture region. Which of those applies is inside the compiled extension and
cannot be established from here.

## Why it matters beyond this machine

A sibling project serving quantized Qwen under vLLM abandoned the W4A8 tier because it broke CUDA
graph capture in the model runner, and that was attributed to the integration rather than the
kernel — on the evidence that the bare op captured. It does, in decode, where M is 1 to 8. In
prefill with a real batch, `M x K` passes 21.8M easily and the bare op refuses here too.

## Suggested first step

A capture-mode check inside `w4a8_codebook_linear_chunked` that either avoids the offending
operation or fails with a message naming it. Right now the caller gets an invalidated capture and
no indication of which kernel invalidated it.
