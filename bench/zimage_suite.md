# Z-Image on this bench — the whole family, and the answer to "is it all converted?"

**Executed 2026-09-12 on the RTX 3090.** The question that started this was whether Z-Image had
actually been converted end to end. **It had not.** Only the third-party `beyond-reality-zimage-v2`
fine-tune had ever been quantized here; both **official** checkpoints had a calibration file on
disk and no output.

That is closed now. Six diffusion builds across three sources, all measured, four newly published.

---

## 1. Every Z-Image checkpoint on this bench

| source | BF16 | our W4A4 | our mixed | published |
|---|---|---|---|---|
| `Tongyi-MAI/Z-Image-Turbo` (official) | 11.46 GiB | **3.06 GiB** | **3.18 GiB** | [Z-Image-Turbo-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Z-Image-Turbo-W4A4-ConvRot) |
| `ostris/Z-Image-De-Turbo` | 11.46 GiB | **3.06 GiB** | **3.17 GiB** | [Z-Image-De-Turbo-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Z-Image-De-Turbo-W4A4-ConvRot) |
| `Beyond_Reality Z-Image v2` | 11.46 GiB | 3.06 GiB | 3.18 GiB | [Beyond-Reality-Z-Image-v2-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Beyond-Reality-Z-Image-v2-W4A4-ConvRot) |

Text encoder `qwen_3_4b` is published separately as
[Qwen3-4B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen3-4B-W4A4-ConvRot) (7.49 → 2.4 GiB).
The VAE (`ae.safetensors`, 320 MiB) is **deliberately not quantized**.

All four new files were verified **byte for byte against the Hub after upload** — an upload that
finishes without an error is not proof that the file on the other side is the file on disk.

Builds that exist only as sidecars, because the file was deleted after being measured:
`zimage-v2-teto-cg16/64/256`, `zimage-v2-mixed-t0.05/0.10/0.20`, `zimage-v2-sigma-*`. Those were
experiments, not deliverables.

---

## 2. The two official checkpoints, measured

Per-layer error over 170 layers, cg 256, against a float32 reference on real sampled activations:

| checkpoint | `err_bf16` | `err_w4a4` | `err_w4a8` |
|---|---|---|---|
| `z_image_turbo_bf16` | 0.0019 | **0.1228** | 0.0389 |
| `z_image_de_turbo_v1_bf16` | 0.0019 | **0.1211** | 0.0384 |

Quality: 6 prompts × 2 seeds × 3 arms per source = **72 renders, none broken.** 8 steps,
1024×1024, euler/simple, cfg 1.0. Sheets: `zimage_turbo_qualidade_s1.png`,
`zimage_deturbo_qualidade_s1.png`.

| source | arm | divergence vs BF16 | spread | s/step | GiB |
|---|---|---|---|---|---|
| Turbo | BF16 | — | — | 0.903 | 11.46 |
| Turbo | W4A4 | 0.6589 | 0.5428 | **0.341** | 3.06 |
| Turbo | mixed | 0.5500 | 0.5003 | 0.391 | 3.18 |
| De-Turbo | BF16 | — | — | 0.911 | 11.46 |
| De-Turbo | W4A4 | 0.4126 | 0.4179 | **0.348** | 3.06 |
| De-Turbo | mixed | 0.4325 | 0.3723 | 0.398 | 3.17 |

**2.65x faster per step and 3.75x smaller**, on both.

**Dispatch counted on all four, weights on the card:** 170 quantized modules each, 8/8 quantized
forwards, **0 `dequantize`**, `convrot_linear_dtype=int4` on `comfy_kitchen.backends.cuda`.

---

## 3. The finding worth keeping: the mixed recipe does not transfer between a model and its own fine-tune

```
paired, run by run, against that source's own W4A4 build
Turbo      mixed   mean delta -0.1089   wins 11/12    <- clearly better
De-Turbo   mixed   mean delta +0.0199   wins  7/12    <- split decision, no winner
```

Same converter, same `--promote-error 0.15`, same six prompts, same two seeds, same card. On the
official Turbo the mixed build is measurably closer to the BF16 trajectory; on its own de-distilled
fine-tune the same recipe buys nothing measurable for 0.11 GiB and 14% more time per step.

**Nothing in the per-layer error predicted that.** The two checkpoints measure 0.1228 and 0.1211 —
within 1.4% of each other — and the promotion picks a near-identical number of layers (57 against
54). Whatever separates them is not in that number.

And a reminder about how to read those means at all: the between-run spread **inside** one arm is
0.37–0.54, which on De-Turbo is **larger than the 0.02 gap between the arm means**. The paired row
is the evidence; the mean is not.

---

## 4. Where Z-Image's ceiling is, and it is the only family here with both ends measured

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| **Z-Image** | **~6 B** | **0.1421** | **0.1848** |
| Krea 2 Turbo | 12.82 B | 0.1377 | not reached on the only axis available |
| HunyuanVideo 1.5 | ~13 B | 0.1837 | 0.2147 |

Filled on 2026-09-03 by pushing `convrot_groupsize` down: cg 256 (0.1312) good, cg 64 (0.1421)
good, **cg 16 (0.1848) destroyed in 3 of 3**. At 0.1211–0.1228 these builds sit well inside the
working band, with room that Krea 2's ceiling probe later showed is not guaranteed to exist in
every model.

---

## 5. A tool defect these conversions exposed

The activation-capture tool read an entire prompt file as **one prompt**: six prompts became a
single six-line prompt and the header announced `encoding 1 prompt(s)`. It did not fail — it
calibrated on one conditioning instead of six, which is worse than failing, because the number
comes out and looks fine. The identical defect had been found and fixed in `quality_ladder.py`
earlier the same day; this was the sibling left behind.

**These builds were produced before that fix**, with the single concatenated prompt — and so was
the published `beyond-reality-zimage-v2` pair they are compared against, so the family stays
internally consistent. A fresh calibration would move the medians by a few percent; the calibration
*seed* alone already moves them 2–6%.

The old `xfer_*` analyses that were going to be reused for these conversions were **refused** by
the converter: they carry no `source_identity_sha256`, written only since 2026-08-22. That is the
guard working — "nothing to check" is not "checked" — and recalibrating cost 83 seconds per
checkpoint.

---

## 6. What none of this covers

- **Six prompts, two seeds, one card, one resolution, no perceptual metric.** "72 of 72 usable" is
  a human judgement over two contact sheets.
- **De-Turbo was sampled at 8 steps / cfg 1.0**, the Turbo regime, for comparability. That is not
  this model's own regime — it exists precisely to undo the distillation.
- **LoRA training on De-Turbo, which is its whole reason to exist, was not tested** on quantized
  weights.
- **No image-space metric and no matched-input probe.** Latent divergence measures trajectory, not
  fidelity.
- **ControlNet, img2img and the Fun-Controlnet-Union patches were not exercised**, on any build.
- **The text encoder was not re-measured here.** `qwen_3_4b`'s quantized build predates the
  text-encoder lock release, so its published numbers describe the dequantized path; see
  [`cadeado_text_encoder.md`](cadeado_text_encoder.md).
