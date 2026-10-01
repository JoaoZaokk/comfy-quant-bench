# Conversão, verificadores e avaliação

> Referência preservada do CLAUDE.md original, linhas 837–1025, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Quantization pipeline

Three tools in `tools/`, all run with the embedded interpreter from the root:

```bash
.\python_embeded\python.exe .\tools\quant_audit.py
```

Recursively walks `ComfyUI/models` (metadata only — parses Safetensors headers, uses the installed GGUF reader for GGUF, safe/meta loading for `.pt`) and rewrites `quantization_inventory.json` + `quantization_inventory.md`. Rerun after producing any new output so the inventory stays current.

```bash
.\python_embeded\python.exe .\tools\quant_w4a4.py --input <src.safetensors> --profile gemma --dry-run
```

Drop `--dry-run` to write. Output name defaults to `<stem>_w4a4_convrot.safetensors` plus a `.quant.json` sidecar manifest.

```bash
.\python_embeded\python.exe .\tools\verify_w4a4.py <out.safetensors> --source <src.safetensors> --kernel-smoke
```

### Mixed precision, calibrated on real activations

A second pipeline that picks 4-bit or 8-bit **per layer** by measuring the real kernels on the
activations the model actually produced. Three tools, in order — see `W4A4_PROGRESS.md` part 9 for
the measurements and `W4A4_HANDOFF.md` for the invocation.

`tools/to_native.py` rewrites a diffusers-named checkpoint under ComfyUI's own module names.
**Not optional for Z-Image.** ComfyUI fuses `attention.to_{q,k,v}` into `attention.qkv` at load
(`comfy/model_detection.py:1498`), and only `.weight` is in that map — `weight_scale`,
`weight_s_rel` and `comfy_quant` pass through unrenamed, so the layer loads with no scale and no
error. The remap is derived from `comfy.utils.z_image_to_diffusers` and checked against ComfyUI's
own `convert_diffusers_mmdit` on meta tensors; its output is bit-identical in the latent.

`tools/calibrate_activations.py` runs real sampling with forward-pre-hooks on every candidate
Linear and reservoir-samples the real input rows (Algorithm R, so the sample is not just step 0).
Samples are stored **bfloat16, never fp16**: real Z-Image activations reach 344064 and fp16 caps
at 65504, and an `inf` there makes every error `nan`, which silently routes the worst layer in the
model to the cheapest format.

`tools/quant_mixed.py` measures `err_bf16` / `err_w4a4` / `err_w4a8` per layer against a float32
reference on those activations, then assigns `convrot_w4a4`, `asym_w4a8_int8`, or no quantization.
Mixed formats in one file are native behaviour: `comfy/utils.py:1454` turns each
`_quantization_metadata["layers"]` entry into a per-layer `<layer>.comfy_quant` tensor and
`comfy/ops.py:1142` dispatches on that layer's own JSON.

Do not choose precision by crest factor. Measured over 170 Z-Image layers its rank correlation
with the W4A4 error is **+0.10**. What correlates is the W4A8 error (+0.978).

### How the converter is built (and why)

- **Native-backend preflight.** Before touching a single tensor, `normal_comfy_backend()` spawns a subprocess that imports `comfy.quant_ops` and asks `comfy_kitchen.registry` which implementation would be selected. If either `quantize_convrot_w4a4_weight` or `convrot_w4a4_linear` does not resolve to `comfy_kitchen.backends.cuda.*`, the conversion **hard-refuses**. Keep this — it is the guard against silently producing a checkpoint that only ever runs dequantized.

  **This paragraph has now been wrong twice, in opposite directions, and the second time is the instructive one.** It first claimed `verify_w4a4.py` ran the same check; it did not. It was corrected on 2026-08-18 to say that `quant_w4a4_smooth.py` and `quant_int8.py` run **no** preflight and that smooth writes no `backend` sidecar field. All three of those became false in `d14ae48` and the paragraph did not follow — so for four days this file told readers that two converters were **less safe than they are**, which is the direction that gets acted on. Re-read against the tree on 2026-08-22: `quant_w4a4_smooth.py:182-191` imports `normal_comfy_backend` and hard-refuses, `:328-330` writes `backend` and `backend_linear`, and `quant_int8.py:187-198` preflights when `--device cuda --convrot` and prints why the `--no-convrot` path is exempt instead of pretending to.

  What is still true, and is the part worth keeping: **the converters do not all resolve the same thing.** `verify_w4a4.py` resolves only `convrot_w4a4_linear` while `quant_w4a4.py` resolves both ops. Do not treat "the tool ran" as proof the CUDA backend was used; check the `backend` field in the sidecar. See [AUDITORIA_2026-08-18.md](../../AUDITORIA_2026-08-18.md) items 8 and 17, and `.scratch/varredura-2026-08-22/issues/05`.

  **The "six definitions" half of this paragraph is fixed and this text was stale about it.** It used to say there were **six** definitions of `normal_comfy_backend` with four different answers, and told the reader to grep and count. Counted 2026-09-01: **zero** — the name is gone entirely, and there is exactly **one** definition of the probe, `native_backend_ready` in `tools/_native_probe.py:315`. `tools/test_native_probe.py` enforces it mechanically (`test_one_definition_of_the_probe_remains`, assembling the needle from two string pieces so its own source is not counted); executed here, **20/20 pass**. A doc that sends someone hunting for six functions that no longer exist costs them the same hour whether the claim was too harsh or too kind.

  **That adoption is DONE and verified on the card — six of seven writers by re-running them, and the numbers are byte-identity, not a code read.** `tools/_conversion.py` holds the write contract once (atomic `.partial` + fsync + `os.replace`, the disk/RAM guards, the refusals) and all seven writers go through it. Executed 2026-09-01 on the 3090, reconverting to a fresh path and comparing the whole file:

  ```
  quant_w4a4    hunyuanvideo1.5 fp16      8 507 690 240 B   sha256 IDENTICO
  quant_w4a8    hunyuanvideo1.5 fp16      8 847 567 376 B   sha256 IDENTICO
  quant_mixed   wan2.1 vace 1.3B          2 310 437 144 B   sha256 IDENTICO
  quant_mixed   beyond-reality-zimage-v2  3 403 133 032 B   sha256 IDENTICO
  to_native     beyond-reality-zimage-v2       11,46 GiB    sha256 IDENTICO
  ```

  `quant_int8` and `svdq_to_bf16` have **no earlier output on disk**, so their acceptance is weaker and says so: a real conversion plus a load through ComfyUI's normal loader. int8 counts 432 `int8_tensorwise` modules with **8/8 quantized forwards, 0 dequantize** on `comfy_kitchen.backends.cuda`; the recovered BF16 loads as `Lumina2`, 6 154 908 736 params, 34 fused qkv keys — for *this* architecture the docstring's fused-layout warning does not bite, because ComfyUI's own `Lumina2` wants `attention.qkv`.

  **`quant_w4a4_smooth` ran end to end on 2026-09-01, and verifying it found two defects in the VERIFIER.** The owner said to stop listing the blocker and download the model. `DreamFast/gemma-3-12b-it-heretic`, matched by **exact size** rather than by name — 23 545 681 250 bytes, the number the w4a8 sidecar already recorded. One conversion then unblocked both missing inputs: `quant_w4a4 --profile gemma` over the BF16 produced the `--calibrate-with`. Smooth wrote 6.91 GiB in 23.3 s, 96/96 norms observed, and the channel outlier ratio went **82.52 → 9.60**.

  Its acceptance then **failed the file twice, and both times the verifier was wrong** — which is the part worth keeping, because both defects had existed for as long as `smooth` has and could only surface by running it to completion:

  - **`Source comparison` reported 96 corrupted tensors** — exactly 48 `input_layernorm` + 48 `pre_feedforward_layernorm`, and nothing else. Rewriting the norm *is* the SmoothQuant mechanism (`norm ← (norm+1)/λ − 1`, `W ← W·λ`), so "preserved == byte-identical" is true for `quant_w4a4` and false by construction here. The fix is not an exemption: for those norms the check **inverts** — they must have changed. A byte-identical norm in a SmoothQuant file means λ=1 there, i.e. the smoothing did nothing, and the file still loads and dispatches.
  - **`--kernel-smoke` failed at rel-RMSE 16.21 against a 0.90 ceiling.** The smoke compares against `F.linear(x, W_source)` while the file stores `W·λ` — two different functions, not two implementations of one. **Tested before fixing**, recovering λ from the converter's own formula inverted (`λ = (norm_src+1)/(norm_out+1)`, nothing stored):

  ```
  λ recovered                              min 4.90  max 103.47  mean 13.60  (3840 channels)
  reference F.linear(x, W_source)          rel-RMSE 15.3650   <- failed
  reference F.linear(x, W_source · λ)      rel-RMSE  0.2127   <- Gemma's normal band
  ```

  72x. Fixed, and λ is now **printed in the report** (min/max/mean) so a reader can check the correction instead of trusting it. Final: structure PASS, source comparison PASS, backend `comfy_kitchen.backends.cuda`, `relative_rmse 0.18702`. `o_proj` and `down_proj` stay outside the correction on purpose — they are quantized but not smoothed, since no norm feeds them directly, and the function returns `None` for them so the strict comparison still applies.

  **A verifier that fails a correct file teaches people to switch verifiers off** — the same argument this repo makes about a WARN resting on a hypothesis. The migration created neither defect; it created the occasion to find them.

  Still open, and now cheap: nothing compared the `_w4a4_convrot` and `_w4a4_smooth` files against each other, so the question smooth exists to answer — *is channel smoothing the largest term of the SVDQuant recipe?* — is unanswered with both files sitting on disk.

  **The negative control in that converter's own refusal test is what found two real defects**, which is the reusable part. The control exists so a converter that died on every invocation could not pass all the refusal cases; it failed, and the failure was the finding. `smooth` had **no zero-layer guard** — pointed at a Wan it printed `Layers: 0 quantized: 0` and exited **0**, so `--dry-run`, the thing you run *before* spending hours, answered SUCCESS for a conversion with nothing to convert (the other four already refused: `quant_w4a4.py:386`, `quant_w4a8.py:248`, `quant_int8.py:139`, `quant_mixed.py:581`). And it had **no dtype guard** — it validated names only, then crashed ~3 minutes later inside another file's `read_tensor`; it now refuses in **1.8 s**, and `tools/test_smooth_guards.py` (7/7) asserts that *time*, because the regression to catch is moving the check back after calibration.

  One thing only the run teaches: **`convrot_groupsize` must be a power of 4, not of 2.** 128 raises `Regular Hadamard size must be a power of 4` at `comfy_kitchen/tensor/int8_utils.py:22`, which is why 64 and 256 are the only values in this tree. The backend preflight caught it before any work.

  Not covered: the three Z-Image builds from before 2026-08-22 are **not reconvertible** — their analyses carry no `source_identity_sha256` and `quant_mixed` refuses rather than skipping the check. That is the guard working; "did not reconvert" is not "reconverted and differed".

  **Measured 2026-08-22, on the 3090, so nobody re-derives it:** the resolved implementation is *invariant* to `convrot_groupsize` and to dummy-vs-real probe tensors. All four combinations — cg 64 and 256, `torch.empty` and real quantized tensors — resolve to `comfy_kitchen.backends.cuda`, and both real calls succeed.

  **That last clause is true of W4A4 and false of W4A8, and the paragraph did not separate them.** Measured 2026-09-01 with `tools/probe_convrot_groupsize.py`, K=2048 so no failure below is about divisibility:

  ```
  cg     quantize_convrot_w4a4_weight   convrot_w4a4_linear   quantize_w4a8_int8_weight
  16     ok                             ok                    RuntimeError
  64     ok                             ok                    RuntimeError
  128    ValueError (power of 4)        --                    RuntimeError
  256    ok                             ok                    ok
  512    ValueError (power of 4)        --                    RuntimeError
  1024   ok                             ok                    RuntimeError
  ```

  So **W4A4 accepts 16 / 64 / 256 / 1024 end to end — quantize *and* execute — while W4A8 accepts only 256.** The error the W4A8 path raises is `convrot rotate kernel only supports group_size 256`, which names no format and reads as a property of the ConvRot kernel; it is not. I was one paragraph away from recording "256 is the only usable value" until the probe refuted it.

  The practical consequence: `quant_mixed` measures **both** formats per layer to choose between them, so it touches the W4A8 path even when the result will be 170/170 in W4A4 — which pins it to cg 256. `quant_w4a4`, which would accept 1024, has no `zimage` profile. That is why the Z-Image ceiling could not be probed on this axis; see `bench/criterio_teto_zimage.md`. So `quant_w4a4.py`'s hardcoded 64/64 preflight against a 256 conversion is untidy, **not** wrong; and `_native_probe.py`'s own docstring claim that dummy kwargs let the check pass where a real call would not **did not reproduce** for these two ops on this build. That is "did not reproduce under the only conditions anyone has tried", not "is false" — the mechanism at `registry.py:246` may still bite elsewhere. `tools/probe_backend_resolution.py` re-runs it.
- **Streaming writes, never mmap.** Output header offsets are computed up front, then tensors are streamed: quantized layers are read by byte range → CUDA → `ck.quantize_convrot_w4a4_weight` → written; everything else is `copy_range`'d verbatim in 16 MiB chunks. **Do not reintroduce `safe_open` / mmap for large sources.** On this Windows host mapping the 21.93 GiB Gemma source failed with `os error 1455` and twice crashed `torch_cpu.dll` with `0xc0000005`.
- **Atomic output.** Writes go to `<output>.partial`, then `os.replace`. Refuses stale partials, refuses existing outputs/sidecars, refuses a source that already has `_quantization_metadata`.
- **Profiles are strict allowlists**, not heuristics. `PROFILE_PATTERNS` matches only `model.layers.N.self_attn.{q,k,v,o}_proj.weight` and `model.layers.N.mlp.{gate,up,down}_proj.weight`; embeddings, norms, `lm_head`, and vision towers are excluded. Only `gemma` and `qwen` exist today. **Do not extend a profile to a new architecture without confirming that architecture's loader and layer config** — Flux, Hunyuan, SeedVR2, and Z-Image each need their own recipe.
- **Group sizes.** `CONVROT_GROUP_SIZE = 256` (rotation), `QUANT_GROUP_SIZE = 64`. Layer selection requires `shape[1] % 256 == 0`. The backend-probe subprocesses in both tools use dummy `64/64` values only to resolve the implementation, which is why they differ from the real conversion values — not a bug, but don't copy those numbers into real calls.

### Output format

Standard Safetensors. Per quantized layer: `<layer>.weight` as `I8` of shape `[rows, cols/2]` (INT8 container holding signed INT4), plus `<layer>.weight_scale` as `F32` of shape `[rows]`. Everything else preserved byte-for-byte. `__metadata__` carries `_quantization_metadata` (`format_version` 1.0 + per-layer `{format: convrot_w4a4, convrot_groupsize: 256}`) and `quantization: "ConvRot W4A4"`. See [ComfyUI/QUANTIZATION.md](../../ComfyUI/QUANTIZATION.md) for the upstream `QuantizedTensor` / `Layout` / `MixedPrecisionOps` model this format plugs into.

**Does `.backends.cuda` in `__module__` prove the native path ran? Measured 2026-08-22: yes, here.** The whole preflight rests on that string match, and the test that settles it is cheap: W4A4 quantizes the **activation** to 4 bits too, so a dequantized-weight fallback (`F.linear(x, W_deq)`, which is what `comfy_kitchen/tensor/convrot_w4a4.py:237` does under one condition) must agree with a real call. It does not — `native` vs `W4-only` is **1.43e-1**, not 1e-6, on a `[1024, 1024]` bf16 weight at cg=256. The A4 half is real. Re-run with `tools/probe_backend_resolution.py`.

`verify_w4a4.py` checks, in order: metadata structure and per-layer dtype/shape → packed shape vs source shape → **byte-identical comparison of every preserved tensor against the source** → native backend resolution → optional real-kernel smoke against `F.linear` on the BF16 source. The smoke's relative RMSE on random inputs (~0.25 for Gemma) is a liveness signal, **not** a quality metric.

`inspect_quant.py <file>` at root is a quick header dump (tensor count, dtype histogram, metadata, scale-like keys).

**There are TWO dialects, and a tool that reads only the first is blind in silence.** The section above describes `__metadata__._quantization_metadata`. A checkpoint may instead ship the per-layer JSON *as tensors* — `<layer>.comfy_quant`, UTF-8 bytes, same content — which is what `comfy/utils.py` produces at load and `comfy/ops.py` dispatches on. Both are valid and a file can carry only the second. Measured 2026-09-01: `LTX25-distilled-DiT-comfy-w4a4` (riftcast, **1440** genuinely 4-bit layers) and both `MiniMax_H3_*_pruned_mixed_int4_int8_convrot` files (117 int4 + 83 `int8_tensorwise`) carry **no `_quantization_metadata` at all**. A reader that checks only the metadata reports **zero quantized layers on a fully quantized file** — and reports it as a clean result, which is the dangerous part. Reading the second dialect costs one seek and ~50 bytes per layer, no torch: see `ler_dialeto_por_tensor` in `tools/avaliar.py`. `quant_audit.py`'s `read_quant_dialects` reads three *metadata* dialects and still does not read this one.

**Batch evaluation, no GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar.py ComfyUI\models --saida .scratch\avaliacao
```

Header + sidecar + `.analysis.json` only — **163 checkpoints in 0.68 s**, no torch, no model load. It computes the median effective error entirely offline (the sidecar says which format each layer got; the analysis says that format's measured error on that layer) and reproduces every number this bench has published. Verdicts are `REPROVADO` / `OLHAR` / `SEM VEREDITO`; **`APROVADO` is deliberately absent**, because no cut on either axis separates usable from unusable here — 0.1837 correct against 0.2147 destroyed, 0.7173 fine against 0.8255 destroyed. It rejects, points and predicts. It does not approve.

**Layer 2, does the quantized math actually run TODAY — needs the GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar_despacho.py --controles
```

The sidecar's `backend` field records the **conversion**, not today's load, and comfy-kitchen, ComfyUI and torch have all moved since. `tools/avaliar_despacho.py` loads through ComfyUI's normal path and counts, reusing `probe_quant_dispatch.py` rather than reimplementing the count. Criterion, predictions and three refutation conditions in `bench/criterio_camada2_despacho.md`, written before running; **all three stayed silent**. Executed 2026-09-01 over the 37 checkpoints layer 1 marked as quantized:

```
26  DESPACHA              every diffusion checkpoint on this bench, 0 dequantize
 6  TRAVADO_PELO_COMFY    all six text encoders
 2  SEM VEREDITO          the two Abiray MiniMax -- die in 7-9 s, not a forward
 2  NAO_PROBAVEL          live in `checkpoints/`, which the probe cannot resolve
 1  NAO_DESPACHA          flux-2-klein-base-4b-fp8
```

So the `backend` field still describes today's execution — including **thirteen builds never loaded here before** and three third-party files (riftcast's `LTX25-distilled-DiT-comfy-*`, 1440 layers each; `DasiwaWAN22I2V14BLightspeed`; Winnougan's `minimax_h3_..._w4a8_convrot`).

**`DESPACHA` is not approval.** It answers one binary question — was the kernel called? — and nothing else. On this bench HunyuanVideo 1.5 W4A4 dispatches natively and the render is destroyed. `APROVADO` stays absent from every layer of the evaluator.

**fp8 runs dequantized, and the `impl` counter says so outright.** `flux-2-klein-base-4b-fp8`, a *diffusion* model with `comfy_force_cast_weights=False` — so not the CLIP lock — makes 0 quantized forwards, 8 `dequantize`, and the op literally called is `dequantize_per_tensor_fp8=comfy_kitchen.backends.cuda`. Memory saved, time not. Open, and not worth guessing at: where `full_precision_mm=True` comes from on a diffusion model, given `comfy/ops.py:1667` passes `disabled=` and not `full_precision_mm` on that path.

**There are TWO text-encoder locks, and the rule that knew only one misclassified the very file that established the finding.** This file documents both — `comfy_force_cast_weights` (from `comfy/sd.py:269`) and `full_precision_mm` (hardcoded at `comfy/sd1_clip.py:114` for every text encoder) — and layer 2's first rule asked only for the first:

```
5 encoders                            force_cast {'True': N}                 -> TRAVADO
qwen3vl_32b_minimax_h3-int4_convrot   force_cast {'False': 351}
                                      fpmm       {'True': 350}               -> NAO_DESPACHA
```

That sixth file is the one this bench measured on 2026-08-31 to establish the lock in the first place. `NAO_DESPACHA` reads as a defect in the checkpoint and would send someone to reconvert a public file that is fine. Either lock alone drops the math, so the rule now asks for either and **names which**. And the defect that let the other one hide: the report did not record the field its own verdict rested on — a verdict whose evidence is not in the report is an opinion.

**Layer 3, the reference-arm guard — needs the GPU:**

```bash
.\python_embeded\python.exe -s .\tools\avaliar_referencia.py --modelo <fp16>.safetensors --clip <te>.safetensors --clip-type wan --steps 25 --size 480 --frames 33 --vace-strength 0.0
```

It does not ask whether the image is good. It asks **whether the model responds to its own conditioning** — two unrelated prompts by two seeds on the *unquantized* file, and the statistic is `d_prompt / d_semente`, where the seed distance is the negative control that makes the ratio readable. Criterion and the three refutation conditions were written in `bench/criterio_guarda_referencia.md` **before** measuring. **Executed 2026-09-01, three families:**

```
braco                                     resposta   veredito       previsao   acertou
Wan VACE 1.0  destruido, verdade          0,2477     REPROVADO      < 0,5      sim
Wan VACE 0.0  bom, verdade                0,7911     OLHAR          > 0,8      NAO, por 1,1%
Z-Image v2 BF16       bom                 1,3459     SEM VEREDITO   > 0,8      sim
HunyuanVideo 1.5 FP16 bom                 2,8603     SEM VEREDITO   > 0,8      sim
```

All three refutation conditions stayed silent: **3.19x** separates the broken arm from the nearest healthy one, the reject threshold cleared all three healthy arms, and the destroyed one did not pass. **It would have aborted the Wan run on the first image instead of the fourth.** The mechanism shows raw, not only in the ratio: on the broken arm the prompt moves the latent **0.075** while the seed moves **0.250** — the model generates from noise and ignores what is asked. The `> 0.8` healthy prediction missed by 1.1% and **the threshold was not moved**; a threshold changed after seeing the number it was meant to classify is a description, not a guard. The reject side rests on **one** genuinely broken arm, and the ratio is noisy in absolute value (Hunyuan's two seeds gave 1.87 and 3.85), so what counts is distance from the threshold, not the second decimal.

Two things layer 1 established on its first pass. **The Z-Image card had 0.1241 on the wrong row** — it belongs to `zimage-v2-w4a4` (170 convrot), not to the mixed build, which is 0.0774; confirmed across nine independent calibrations, corrected the same day. The band's `tolerado` value is unchanged, it just gained an owner, and it is the *most aggressive* build measured. And **the calibration seed moves the median 2-6%**: the same Wan checkpoint measures 0.051807 or 0.054631 depending on which calibration you use. Smaller than the band's own 45% width, so the per-model line survives — but a four-decimal number from one calibration claims precision this bench does not have, so the spread now travels beside it.

