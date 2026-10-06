# Kernel optimization on the RTX 3090: Qwen-Image-2.1, 2026-10-01 .. 10-06

Seven phases of work on making the quantized Qwen-Image-2.1 path in ComfyUI faster and more faithful on an RTX 3090
(sm86, power-limited to 280 W). The phases started out reproducing the *spatial* tricks of a commercial W4A4 engine
(group size, rotation, low-rank branch) and ended on a *temporal* lever that is much larger: **where in the denoise the
precision is spent**.

Every number below is measured on this bench unless tagged otherwise. Fidelity is PSNR of the RGB output against the BF16
model with the same prompt, seed and graph (12 prompts × 2 seeds = 24 graphs, or × 6 seeds = 72 graphs), paired
cluster bootstrap for the confidence intervals (`experiments/qwen21-kernel/analysis/estat_bateria.py`). Time is
the KSampler time, 1024², 25 steps, euler/simple. "FA2" = `--use-flash-attention` (quality mode of the launcher);
"fast" = SageAttention with `COMFY_SAGE_PV_ACCUM=fp16sv` (image mode of the launcher). Criteria were written before each
measurement and amended only with a dated note.

## Result (phase 7, 72 graphs)

| configuration | PSNR FA2 | time FA2 | PSNR fast | time fast | peak VRAM |
|---|---:|---:|---:|---:|---:|
| W4A4 ConvRot | 18.92 dB | 9.75 s | 18.89 dB | 8.17 s | ~12.4 GB |
| **INT8 first 3 steps → W4A4** | **23.92** | **9.97** | **23.85** | **8.67** | ~19.3 GB |
| **INT8 first 5 steps → W4A4** | **25.44** | **10.73** | **25.25** | **9.07** | ~19.3 GB |
| p008 mixed (157 INT8 + 35 W4A4 layers) | 25.46 | 13.90 | 25.38 | 12.46 | ~15.4 GB |
| **INT8 ConvRot, all layers** | **32.68** | **14.52** | **31.51** | **12.78** | ~15.7 GB |
| own g128/g128 W4A4 kernel (the original goal) | 20.78 | 11.42 | — | — | ~12.4 GB |

- **Quality:** full INT8 ConvRot is +7.2 dB over p008 for +4.5 % time at the same VRAM. The loss of p008 came from its
  35 W4A4 layers.
- **Speed:** INT8 for the first 5 steps then W4A4 matches p008 fidelity in 23–27 % less time; 3 steps buys +5 dB over
  plain W4A4 for +2–6 % time.
- **Mechanism:** the divergence from the BF16 trajectory is born in the first steps; later steps do not repair it.
  Spending INT8 at the end instead ("sandwich" INT8/W4A4/INT8) is worse than spending the same steps at the start.
- **Implementation:** two stock `KSamplerAdvanced` nodes (`start_at_step`/`end_at_step`, leftover noise passed through)
  with two `UNETLoader`s. Same result as switching inside a custom node (0.08 dB, not significant).

Curve, INT8 for the first *k* of 25 steps (24 graphs, FA2):

| k | 0 | 1 | 2 | 3 | 5 | 8 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR (dB) | 18.8 | 20.6 | 22.2 | 23.5 | 25.1 | 26.3 | 27.0 | 28.1 | 29.1 | 32.5 |
| time (s) | 9.72 | 9.61 | 9.79 | 10.00 | 10.68 | 10.82 | 11.62 | 12.14 | 13.02 | 14.19 |

**Not yet validated:** other resolutions and aspect ratios, other step counts, samplers and schedulers, image edit,
LoRA, and a blind visual review. *k* is only meaningful for 25 steps of euler/simple; the transferable quantity is the
sigma region of the schedule, which is the next measurement.

## Modules

| module | what it is | where |
|---|---|---|
| fused SwiGLU W4A4 quantizer | SwiGLU folded into the ConvRot activation quantizer (−0.60 ms/layer; Qwen W4A4 7.25 → 6.76 s) | `tools/ck_swiglu/`, `patches/comfy_kitchen_swiglu_w4a4_fused.patch`, `patches/comfyui_swiglu_triton_env.patch` |
| Sage policy per model + V-scaled fp16 PV | `COMFY_SAGE_PV_ACCUM=fp16sv` without black images; per-model `smooth_k`/per_warp | `patches/comfyui_sage_pv_accum_env.patch` |
| mixed-precision converter | promotes W4A4 layers above an error threshold to INT8 ConvRot (p008/p010/...) | `tools/quant_mixed.py` |
| GPTQ for W4A4 | integer replacement with a large Hessian; +0.14..+0.37 dB at n=72, not a default | `tools/gptq_w4a4.py` |
| `g64_gemm6` | own IMMA m16n8k64 s4 kernel, row/row and per-group (g64/g128) scale policies | `experiments/qwen21-kernel/kernels/g64_gemm6.cu` |
| `g64_quant`, `g64_quant2` | Hadamard-256 + per-group activation quantizer, optional clip ratio | `experiments/qwen21-kernel/kernels/g64_quant*.cu` |
| g64 test node | loads the kernels into ComfyUI; per-layer policy lists, `G64_TROCA` in-node step switch | `experiments/qwen21-kernel/node/` |
| attention emulation | INT4/FP4/INT8 QK and PV error on a real Qwen attention dump | `experiments/qwen21-kernel/analysis/attn_int_emul.py` |
| step-switch workflows | INT8 → W4A4 at step 3 / 5 with stock nodes | `experiments/qwen21-kernel/workflows/QWEN21-TXT2IMG-int8-w4a4-k{3,5}.json` |

## The phases

| phase | date | outcome |
|---|---|---|
| 1 | 10-01/02 | Black images = fp16 overflow of Sage's PV accumulator; fixed by scaling V per column. Sage's INT8 QK errs 10–90 % on heads with a dominant coordinate; `smooth_k` decides the sign per model. Largest cost on the bench was configuration (`--cache-none`, VAEs on the NAS), not kernels. |
| 2 | 10-02 | Launcher on `fp16sv`. Mixed W4A4+INT8 first measured. FP8 / Sage2++ does not exist on sm86. |
| 3 | 10-02 | Audit of phase-2 negatives: three were implementation or calibration, not format. Mixed policy sweep (p015..p008). Chaos floor: two exact attention kernels differ by ~30 dB, 1 ulp by ~28 dB. |
| 4 | 10-03 | Fused SwiGLU and per-model Sage policy shipped. GPTQ not significant at n=72. p008 over p012 by +2.77 dB. FA3 on sm86 = FA2. |
| 5 | 10-03 | Quality mode on FA2 (−2.5..−4.2 %). nsys profiles: activation quantizer at 37–61 % of bandwidth. |
| 6 | 10-04 | Commercial r32 format decomposed in images: the whole gain is g64 scales on weights and activations (+2.58 dB); rotation variants, asymmetry, low-rank branch on top: +0.52 dB, CI includes 0. |
| 7 | 10-05/06 | Real g64/g128 kernel built (+1.85 dB at +17 %), then overtaken by full INT8 and by the INT8→W4A4 step switch (table above). |

## Closed negatives (measured)

- g256 groups; g128 on a subset of layers (gate_up +0.00 dB, to_v/to_out +0.38 n.s., mlp.out −0.32 n.s.).
- MSE-optimal weight scales (−1.13 dB) and activation clipping (+0.10 / −0.25 dB, n.s.): per-layer output error fell
  5–8 % and the image did not improve. Per-layer error does not approve a format.
- INT8 at the end of the trajectory (sandwich) instead of the start.
- 4-bit QK attention: INT4 in blocks of 16 has 11× the error of INT8; NVFP4/MXFP4 are worse than INT4.
- FlashAttention 4 (4.0.0b33) on sm86 runs the SM80 path, the FA2 algorithm in CuTe DSL: 4.6–5.0 % slower than FA2,
  same error. SageAttention 3 is FP4 for sm100/sm120/sm121; Sage2++ gains come from FP8 (sm89+).
- [HARDWARE LIMIT] At 280 W the INT4 GEMM is power-bound: per-group scale policies cost energy, tile configs are neutral
  on FA2 (±2 %).

## Backlog (not measured, low ROI for now)

Deprioritized, not refuted. Anything orthogonal to the step switch still composes with it.

- **Single INT8 copy for the step switch.** The ~19.3 GB peak is the main defect of k3/k5. Two options: derive W4A4
  from the resident INT8 once at the switch (measure the repack time and the fidelity change against W4A4 derived from
  BF16), or store INT8 as two INT4 halves read by one kernel (−3.7 GB, needs a kernel).
- **QFA / attention time** on full INT8 or k5 without changing the trajectory.
- **INT8 PV in Sage:** lossless in emulation (+5 % error with per-block scales), ceiling ~3 % in fast mode.
- ResQ / r32 selective, V-scaling fusion, AutoRank, HIGGS/HQQ, joint optimization: ceilings never measured.
- Nsight Compute: blocked on GPU performance-counter access (owner's system setting).

## Next

1. Blind visual review: BF16 / INT8 / k5 / p008 side by side on the problem prompts.
2. Switch boundary as a sigma region: short batteries at 40 steps, other schedulers, 1328² / 1536² / 2048², image edit.
   Widen to 72 graphs only where the curve moves.
3. Full INT8 as quality mode; k3 / k5 as `fast` / `balanced` presets on stock nodes.
4. Remove the duplicated weights (backlog item 1).
5. Then orthogonal speed work (QFA, Nsight).

## Sources

Code: [`experiments/qwen21-kernel/`](../experiments/qwen21-kernel/). The bench's working notes, battery harness and raw logs are kept locally and not published;
every number above comes from them.
