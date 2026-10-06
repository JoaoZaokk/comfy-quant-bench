# Kernel optimization on the RTX 3090: Qwen-Image-2.1, 2026-10-01 .. 10-06

Eight phases of work on making the quantized Qwen-Image-2.1 path in ComfyUI faster and better on an RTX 3090 (sm86,
power-limited to 280 W). They started out reproducing the *spatial* tricks of a commercial W4A4 engine (group size,
rotation, low-rank branch), found a large *temporal* effect in PSNR (where in the denoise the precision is spent), and
then a blind visual review showed that PSNR had been measuring the wrong thing: **what the eye sees is decided by which
layers stay in 4 bits, not by when.**

Every number below is measured on this bench unless tagged otherwise. Time is the KSampler time, 1024², 25 steps,
euler/simple. "FA2" = `--use-flash-attention` (quality mode of the launcher); "Sage" = SageAttention with
`COMFY_SAGE_PV_ACCUM=fp16sv` (image mode of the launcher). Criteria were written before each measurement and amended only
with a dated note.

## Result (phase 8): blind review by the owner

Three rounds of 24 scenes (12 prompts × 2 seeds). Every version of a scene shown side by side in a per-scene random order,
labels kept off the page; full-size flicker comparison between versions. One rater (the bench's owner).

**Round 1 — BF16, full INT8, INT8-then-W4A4 (k5), p008** (the 24 scenes where k5 or p008 were furthest from BF16):

| version | picked as worst | picked as best | PSNR vs BF16 |
|---|---:|---:|---:|
| BF16 | 0 | 5 | — |
| full INT8 ConvRot | 0 | 6 | 30.2 dB |
| INT8 for the first 5 steps, then W4A4 | **24 of 24** | 0 | 24.6 dB |
| p008 mixed | 0 | 10 | 22.6 dB |

The step switch was the worst version in every scene although its PSNR is 2 dB *higher* than p008's. PSNR against BF16
measures whether the trajectory, and so the composition, stays the same; it does not measure whether the image is clean.
The step switch repairs composition (decided in the first steps) and leaves the finish (the last steps) in W4A4, which is
what the eye judges. Full INT8 and BF16 were repeatedly called "nearly identical".

**Round 2 — step switches against W4A4 and p008** (grades: good / OK / bad):

| version | bad | good or best | mean score (0–3) |
|---|---:|---:|---:|
| W4A4 | 18 | 0 | 0.25 |
| INT8 first 5 steps → W4A4 | 17 | 2 | 0.38 |
| INT8 last 5 steps only | 13 | 4 | 0.67 |
| INT8 first 5 + last 5 | 8 | 10 | 1.09 |
| p008 | **0** | 18 | 2.08 |

INT8 early is indistinguishable from plain W4A4 to the eye (Wilcoxon p 0.41); no step schedule comes close to p008, which
keeps 35 layers in W4A4 at every step and still never looks bad.

**Round 3 — mixed checkpoints** (W4A4 layers above an error threshold promoted to INT8 ConvRot, `tools/quant_mixed.py`):

| version | W4A4 / INT8 layers | bad | good or best | Sage | FA2 | file |
|---|---|---:|---:|---:|---:|---:|
| W4A4 | 192 / 0 | 18 | 0 | 8.41 s | 9.72 s | 3.77 GB |
| p015 | 120 / 72 | 6 | 11 | 9.25 s | 10.56 s | 4.61 GB |
| p012 | 88 / 104 | 6 | 13 | 10.92 s | 12.07 s | 5.81 GB |
| **p010** | 68 / 124 | **0** | 17 | 11.79 s | 12.89 s | 6.55 GB |
| p008 | 35 / 157 | **0** | 19 | 12.96 s | 13.80 s | 6.93 GB |
| full INT8 ConvRot | 0 / 192 | (never worst, round 1) | — | 12.70 s | 14.19 s | 7.26 GB |

The step in quality sits between p012 and p010: with 88 or more W4A4 layers the same hard scenes (motion, crowds) fail;
with 68 none does, although the owner noted p010 can look slightly forced where even p008 struggles. p010 and p008 are not
separable at n = 24. **In Sage mode full INT8 is as fast as p008**, so p008 is dominated.

**Recommendation:** full INT8 ConvRot (`tools/quant_int8.py --convrot`) — the owner's pick; p010 when the ~7 % of Sage
time matters; p015 only if some failures on hard scenes are acceptable. Limits: 24 scenes, one rater, FA2 renders (Sage
measured for time only).

## Fidelity to BF16 (phase 7, 72 graphs) — composition, not quality

| configuration | PSNR FA2 | time FA2 | PSNR Sage | time Sage | peak VRAM |
|---|---:|---:|---:|---:|---:|
| W4A4 ConvRot | 18.92 dB | 9.75 s | 18.89 dB | 8.17 s | ~12.4 GB |
| INT8 first 3 steps → W4A4 | 23.92 | 9.97 | 23.85 | 8.67 | ~19.3 GB |
| INT8 first 5 steps → W4A4 | 25.44 | 10.73 | 25.25 | 9.07 | ~19.3 GB |
| p008 mixed | 25.46 | 13.90 | 25.38 | 12.46 | ~15.4 GB |
| full INT8 ConvRot | 32.68 | 14.52 | 31.51 | 12.78 | ~15.7 GB |
| own g128/g128 W4A4 kernel (the original goal) | 20.78 | 11.42 | — | — | ~12.4 GB |

The step switch (two stock `KSamplerAdvanced` nodes) does keep the composition closer to BF16, and the boundary is a
region of the noise level, not a step count (phase 8, 24 graphs each, gain over W4A4 at the same settings):

| switch at σ | beta scheduler, 25 steps | simple, 40 steps | simple, 1328², 25 steps |
|---|---:|---:|---:|
| ≈ 0.83 | k5: +5.36 dB | k5: +5.52 dB | k3: +6.01 dB |
| ≈ 0.72–0.73 | k7: +6.84 dB | k8: +6.94 dB | k5: +7.92 dB |

Same σ, same gain, whatever the scheduler or step count; the gain grows with resolution. None of this survives the blind
review above.

## Modules

| module | what it is | where |
|---|---|---|
| fused SwiGLU W4A4 quantizer | SwiGLU folded into the ConvRot activation quantizer (−0.60 ms/layer; Qwen W4A4 7.25 → 6.76 s) | `tools/ck_swiglu/`, `patches/comfy_kitchen_swiglu_w4a4_fused.patch`, `patches/comfyui_swiglu_triton_env.patch` |
| Sage policy per model + V-scaled fp16 PV | `COMFY_SAGE_PV_ACCUM=fp16sv` without black images; per-model `smooth_k`/per_warp | `patches/comfyui_sage_pv_accum_env.patch` |
| mixed-precision converter | promotes W4A4 layers above an error threshold to INT8 ConvRot (p008 / p010 / p012 / p015) | `tools/quant_mixed.py` |
| INT8 converter | int8_tensorwise with ConvRot rotation | `tools/quant_int8.py` |
| GPTQ for W4A4 | integer replacement with a large Hessian; +0.14..+0.37 dB at n=72, not a default | `tools/gptq_w4a4.py` |
| `g64_gemm6` | own IMMA m16n8k64 s4 kernel, row/row and per-group (g64/g128) scale policies | `experiments/qwen21-kernel/kernels/g64_gemm6.cu` |
| `g64_quant`, `g64_quant2` | Hadamard-256 + per-group activation quantizer, optional clip ratio | `experiments/qwen21-kernel/kernels/g64_quant*.cu` |
| g64 test node | loads the kernels into ComfyUI; per-layer policy lists, `G64_TROCA` in-node step switch | `experiments/qwen21-kernel/node/` |
| attention emulation | INT4/FP4/INT8 QK and PV error on a real Qwen attention dump | `experiments/qwen21-kernel/analysis/attn_int_emul.py` |
| step-switch workflows | INT8 → W4A4 at step 3 / 5 with stock nodes (composition only, see above) | `experiments/qwen21-kernel/workflows/QWEN21-TXT2IMG-int8-w4a4-k{3,5}.json` |

## The phases

| phase | date | outcome |
|---|---|---|
| 1 | 10-01/02 | Black images = fp16 overflow of Sage's PV accumulator; fixed by scaling V per column. Sage's INT8 QK errs 10–90 % on heads with a dominant coordinate; `smooth_k` decides the sign per model. Largest cost on the bench was configuration (`--cache-none`, VAEs on the NAS), not kernels. |
| 2 | 10-02 | Launcher on `fp16sv`. Mixed W4A4+INT8 first measured. FP8 / Sage2++ does not exist on sm86. |
| 3 | 10-02 | Audit of phase-2 negatives: three were implementation or calibration, not format. Mixed policy sweep (p015..p008). Chaos floor: two exact attention kernels differ by ~30 dB, 1 ulp by ~28 dB. |
| 4 | 10-03 | Fused SwiGLU and per-model Sage policy shipped. GPTQ not significant at n=72. p008 over p012 by +2.77 dB. FA3 on sm86 = FA2. |
| 5 | 10-03 | Quality mode on FA2 (−2.5..−4.2 %). nsys profiles: activation quantizer at 37–61 % of bandwidth. |
| 6 | 10-04 | Commercial r32 format decomposed in images: the whole gain is g64 scales on weights and activations (+2.58 dB); rotation variants, asymmetry, low-rank branch on top: +0.52 dB, CI includes 0. |
| 7 | 10-05/06 | Real g64/g128 kernel built (+1.85 dB at +17 %), then overtaken in PSNR by full INT8 and by the INT8→W4A4 step switch. |
| 8 | 10-06 | Blind review: the step switch is the worst version in 24/24 scenes; PSNR tracks composition, not quality. Full INT8 for quality, p010 as the cheapest clean mix. Step-switch boundary is a σ region. |

## Closed negatives (measured)

- The INT8 → W4A4 step switch as a speed mode: worst in 24/24 blind scenes, equal to plain W4A4 to the eye. Any step
  schedule (INT8 early, late, or both) stays far behind p008.
- PSNR against BF16 as an acceptance metric for quantized diffusion: it ranked the step switch above p008; the blind review
  ranked it last. Fidelity and quality need separate checks.
- g256 groups; g128 on a subset of layers (gate_up +0.00 dB, to_v/to_out +0.38 n.s., mlp.out −0.32 n.s.).
- MSE-optimal weight scales (−1.13 dB) and activation clipping (+0.10 / −0.25 dB, n.s.): per-layer output error fell
  5–8 % and the image did not improve. Per-layer error does not approve a format.
- 4-bit QK attention: INT4 in blocks of 16 has 11× the error of INT8; NVFP4/MXFP4 are worse than INT4.
- FlashAttention 4 (4.0.0b33) on sm86 runs the SM80 path, the FA2 algorithm in CuTe DSL: 4.6–5.0 % slower than FA2,
  same error. SageAttention 3 is FP4 for sm100/sm120/sm121; Sage2++ gains come from FP8 (sm89+).
- [HARDWARE LIMIT] At 280 W the INT4 GEMM is power-bound: per-group scale policies cost energy, tile configs are neutral
  on FA2 (±2 %).

## Backlog (not measured, low ROI for now)

Deprioritized, not refuted.

- **Which layers fail.** Between p012 and p010, 20 layers decide whether hard scenes fail; naming them could give a mix
  as clean as p010 with fewer INT8 layers.
- **QFA / attention time** on full INT8 without changing the output.
- **INT8 PV in Sage:** lossless in emulation (+5 % error with per-block scales), ceiling ~3 % in Sage mode.
- ResQ / r32 selective, V-scaling fusion, AutoRank, HIGGS/HQQ, joint optimization: ceilings never measured.
- Nsight Compute: blocked on GPU performance-counter access (owner's system setting).

## Sources

Code: [`experiments/qwen21-kernel/`](../experiments/qwen21-kernel/). The bench's working notes, battery harness, blind-review
keys and raw logs are kept locally and not published; every number above comes from them.
