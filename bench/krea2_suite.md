# Krea 2 on this bench — every build, what it costs, and what it is good for

Everything below was **executed on this machine** on 2026-09-12: an RTX 3090, ComfyUI 0.33.0,
comfy-kitchen 0.2.31, torch 2.13.0+cu130. Nothing here is inferred from a code read unless the
line says so. Each number names the run it came from.

Method and the criteria written *before* each measurement:
[`criterio_quant_krea2.md`](criterio_quant_krea2.md) (the conversion),
[`criterio_teto_krea2.md`](criterio_teto_krea2.md) (the ceiling),
[`krea2_edit_verificacao.md`](krea2_edit_verificacao.md) (the edit path).

---

## 1. The builds

| file | format | GiB | median `err_w4a4` | origin |
|---|---|---|---|---|
| `krea2_turbo_bf16` | reference, BF16 + F32 ends | 24.48 | — | `krea/Krea-2-Turbo`, Comfy-Org repack |
| `krea2_turbo_int8_convrot` | `int8_tensorwise` × 224 | 12.57 | — | Comfy-Org, public |
| `krea2_raw_int8_convrot` | `int8_tensorwise` × 224 | 12.57 | — | Comfy-Org, public, **non-turbo base** |
| `krea2_turbo_w4a4` | `convrot_w4a4` × 224 | **7.50** | **0.1199** | ours, cg 256 |
| `krea2_turbo_mixed` | 173 × 4-bit + 51 × 8-bit | **7.70** | — | ours, `--promote-error 0.15` |
| `krea2_turbo_w4a4_cg64` | `convrot_w4a4` × 224 | 7.50 | 0.1238 | ours, ceiling probe |
| `krea2_turbo_w4a4_cg16` | `convrot_w4a4` × 224 | 7.50 | 0.1377 | ours, ceiling probe |

**12,820,073,036 parameters**, summed from the safetensors header rather than estimated from the
file size.

The model is a `SingleStreamDiT`: 28 blocks × 8 Linears = the 224 layers the `krea2` profile
selects. Modulation is an `nn.Parameter`, not a Linear, so it is outside the profile by
construction rather than by a decision. So are the 32 Linears of the `txtfusion` sub-network and
the model's ends (`first`, `last.linear`, `tmlp`, `tproj`, `txtmlp`) — 41 excluded 2-D tensors in
total, of which 39 would pass the divisibility filter. **That exclusion list has never been
tested**; see §6.

The two ceiling probes carry the **same tensor payload** as the shipped build — 8,057,415,984 B
in every case, because the ConvRot scale is `weight_scale` F32 `[rows]`, per row, so the rotation
size does not change how many scales exist. The files differ by exactly 224 bytes, one character
per layer in the per-layer JSON.

---

## 2. Where the error sits, per layer

Measured over 224 calibrated layers against a float32 reference, on the activations the layers
actually received during sampling (`calib/krea2_turbo.calib.pt`, reservoir-sampled across steps).

```
err_bf16    median 0.0019   max 0.0025      <- the floor; this is not quantization
err_w4a4    median 0.1199   p25 0.0822   p75 0.1475   max 0.2942
err_w4a8    median 0.0373   p25 0.0264   p75 0.0467   max 0.0634
```

`err_w4a8 < err_w4a4` in **224 of 224** layers, median ratio **3.12x**.

By family, and the pattern is not random:

| family | n | median | | family | n | median |
|---|---|---|---|---|---|---|
| `attn.wo` | 28 | **0.2181** | | `mlp.gate` | 28 | 0.1181 |
| `mlp.down` | 28 | **0.2029** | | `attn.gate` | 28 | 0.0979 |
| `mlp.up` | 28 | 0.1370 | | `attn.wq` | 28 | 0.0813 |
| `attn.wv` | 28 | 0.1283 | | `attn.wk` | 28 | 0.0749 |

The two worst families are both **output projections** — `attn.wo` and `mlp.down`, the layers that
write back into the residual stream. The two cheapest are the query and key projections. That is
what `--promote-error 0.15` acts on: it promotes 51 layers to 8-bit, and they are overwhelmingly
those two families.

---

## 3. Quality: 40 renders, 4 arms

5 prompts × 2 seeds × 4 checkpoints, 10 steps, 1024×1024, euler/simple, cfg 1.0.
**40 of 40 usable, none broken.** Sheets: `krea2_qualidade_s1.png`, `_s2.png`.

| arm | latent divergence vs BF16 | between-run spread | s/step | GiB |
|---|---|---|---|---|
| `krea2_turbo_bf16` | — | — | 2.239 | 24.48 |
| `krea2_turbo_int8_convrot` | 0.2435 | 0.6512 | 1.233 | 12.57 |
| `krea2_turbo_w4a4` | 0.5843 | 0.7005 | 0.839 | 7.50 |
| `krea2_turbo_mixed` | 0.4972 | 0.7257 | 1.000 | 7.70 |

**Read the paired row, not the means.** The spread *inside* a single arm (0.65–0.73) is larger
than the gaps *between* the arm means, so a mean comparison here is noise with a decimal point on
it. Paired run by run, same prompt and same seed:

```
vs krea2_turbo_int8_convrot    mean delta    wins
krea2_turbo_w4a4                 +0.3408     0/10
krea2_turbo_mixed                +0.2537     0/10
```

**Comfy-Org's own int8 build beats our W4A4 in 10 of 10 paired runs**, 2.40x more faithful to the
BF16 latent. That is the fourth independent measurement on this bench pointing the same way, now
in a fourth model family (Z-Image with real activations 1.49x, epsilon-per-step 1.33x, MiniMax
60/60 at 1.40x).

And our W4A4 is **1.47x faster per step** (0.839 against 1.233) and **1.68x smaller** (7.50 against
12.57 GiB). This is a trade, not a defect. What cannot be said is that W4A4 is more faithful.

**A caveat that belongs on the divergence column specifically.** 0.5843 sounds alarming and is
not: at 10 steps a small perturbation reroutes the sampler, and the place it arrives is a
different good image. This bench measured that directly in August — a generated image cannot
compare two quantizations of the same model, only a matched-input probe can. What the number
measures is trajectory, not damage.

**The reference arm was checked before any of this was believed.** `avaliar_referencia.py` on the
BF16 file gives `d_prompt / d_seed = 1.7686` — it responds to its own conditioning, with the seed
distance as the negative control. The prediction written beforehand was `> 0.8`.

---

## 4. The ceiling: searched for, not reached

Full criterion and the six predictions in [`criterio_teto_krea2.md`](criterio_teto_krea2.md);
**4 confirmed, 2 refuted**.

The format exposes exactly one axis that raises per-layer error: `convrot_groupsize`, legal at
16 / 64 / 256 / 1024, where *smaller* is worse. (A Hadamard rotation of size N spreads each
outlier across N channels, so a bigger N flattens outliers harder. The opposite intuition was
written down on Z-Image in September and measured to be backwards.)

| cg | median | p25 | p75 | max | vs 256 | render |
|---|---|---|---|---|---|---|
| 256 | 0.1199 | 0.0822 | 0.1475 | 0.2942 | 1.000x | 10/10 good |
| 64 | 0.1238 | 0.0886 | 0.1553 | 0.3397 | 1.033x | 10/10 good |
| 16 | **0.1377** | 0.1079 | 0.1961 | 0.4331 | 1.149x | **10/10 good** |

Monotone in the median and in **215 of 224** layers individually. Measured over the intersection —
all three builds select the same 224 layers, so no two populations are compared as one.

**At the smallest legal groupsize the model does not break.** The `"OPEN"` sign is legible in both
cg-16 cells; the portrait keeps skin and wrinkles; the frost macro keeps fine structure. Sheets:
`krea2_teto_s1.png`, `krea2_teto_s2.png`, BF16 in the first row as the control that has to pass.

So the tolerated value rises to **0.1377** and the upper bound stays **unreached** — with a reason
written next to it, not a blank cell.

The obvious wrong reading — *the picture survived because the 4-bit kernel never ran* — was ruled
out rather than assumed:

```
cg 64   224 quantized modules   8/8 quantized forwards   0 dequantize   int4   backends.cuda
cg 16   224 quantized modules   8/8 quantized forwards   0 dequantize   int4   backends.cuda
```

**Two transfer hypotheses died on this checkpoint in one day.** The prediction for the conversion
was that a 12.8 B model would land where the ~13 B HunyuanVideo does (0.15–0.22); it measured
0.1199, *below* the ~6 B Z-Image, killing size monotonicity. The prediction for the ceiling
applied Z-Image's groupsize ratios (1.155x, 1.468x); Krea2 measures 1.033x and 1.149x, three times
less sensitive. Both the tolerance **and** the groupsize sensitivity belong to the model, not to
the format.

Where this leaves the cross-model table:

| model | parameters | tolerated | not tolerated |
|---|---|---|---|
| Wan 2.1 VACE | 1.3 B | 0.0546 | 0.0793 |
| Z-Image v2 | ~6 B | 0.1421 | 0.1848 |
| **Krea 2 Turbo** | **12.82 B** | **0.1377** | **not reached on the only axis available** |
| HunyuanVideo 1.5 | ~13 B | 0.1837 | 0.2147 |

---

## 5. Speed

| arm | s/step | note |
|---|---|---|
| BF16 | 2.239 / 2.284 | 24.5 GiB on a 24 GB card: offloads whole. Not comparable to anything. |
| int8 | 1.233 | |
| mixed | 1.000 | |
| W4A4 cg 256 | 0.839 / 0.914 | |
| W4A4 cg 64 | 0.886 | |
| W4A4 cg 16 | 0.972 | |

Two numbers appear where an arm ran in both ladders, and the gap between them is the point:
**cg 256 measured 0.839 in one run and 0.914 in the next, 9%, with no code change** — the second
ladder shared the disk with a 57 GiB download. The ceiling prediction was that the three
groupsizes would land within 10% of each other; they land within 9.7%. **A 10% effect measured
under 9% contention is not a strong result.** Re-run on a quiet machine before quoting it.

---

## 6. What none of this covers

- **Five prompts, two seeds, one card, one resolution, no perceptual metric.** "40 of 40 usable"
  is the judgement of someone who looked at two contact sheets, not a metric.
- **`cg 1024` was not measured.** It moves toward *less* error; the ceiling run was looking up.
- **The profile's exclusion list is untested.** 39 excluded 2-D tensors would pass the
  divisibility filter — the whole `txtfusion` sub-network, `tproj`, `tmlp`, `txtmlp`,
  `last.linear`. Quantizing them would not move the median (it changes *which* layers degrade, not
  how much), so it answers a different question, and it needs a new calibration because the
  current `.calib.pt` only holds the 224 layers' activations.
- **`krea2_raw` has no BF16 on disk** — only Comfy-Org's int8 build. Nothing here was measured on
  the non-turbo base, which is not distilled and wants ~28 steps and real cfg.
- **The matched-input epsilon comparison never ran on Krea2.** It is the right instrument to
  separate two quantizations of one model, and it is blocked: `comfy_aimdo/host_buffer.py:6`
  freezes `lib = control.lib` at import time, and `comfy/memory_management.py:7` imports it as
  soon as ComfyUI is imported, so the probe's subprocess dies on a `None` lib. Without DynamicVRAM
  the BF16 arm commits 49.4 GiB of pagefile at 4 percent CPU. An attempted fix was reverted
  because it did not work and would have changed the load path for Z-Image, where the probe works
  today.
- **The text encoder is still BF16 and still locked.** ComfyUI runs every text encoder with
  dequantized math behind two independent locks (`comfy_force_cast_weights` from `comfy/sd.py:269`
  and `full_precision_mm` hardcoded at `comfy/sd1_clip.py:114`). Quantizing it saves VRAM and no
  time unless both are released, which no node exposes.

---

## 7. One trap this model taught the bench

`krea2_turbo_bf16` is the first checkpoint here with **two floating dtypes among its 2-D weights**:
the 256 `blocks` and `txtfusion` weights are BF16, and the ends — `first`, `last.linear`, `tmlp`,
`tproj`, `txtmlp` — are F32. That is the model's own precision policy, not a defect.

With DynamicVRAM on, `comfy/ops.py:552` assigns a Linear's weight **straight from the file and
never casts it to the module's dtype**. After reading 24.5 GiB, the run dies with

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
  comfy/ldm/krea2/model.py:331, img = self.first(img)
```

which names the first layer of the model and says nothing about the mechanism. `disable_dynamic=True`
on the loaders does **not** protect you, because `tools/_dynamic_vram.enable()` turns aimdo on
globally before either call. And turning DynamicVRAM off trades a dtype bug for a memory one:
19.7 GiB of pagefile with 8.5 GiB resident, 7521 page faults in 6 s, 0.03 s of CPU in the same
interval.

The fix in `tools/quality_ladder.py` is to cast after loading — `casta_pesos_divergentes()`, 15
tensors — which only reaches the state the non-lazy path already produced on its own.
`tools/_dynamic_vram.perigoso_para_lazy()` answers it from the header alone, before any load.

---

## 8. The text encoder, which is where the next win actually is

After the DiT drops to 7.50 GiB, **the text encoder is the largest file in the Krea 2 pipeline**:
`qwen3vl_4b_bf16` is 8.27 GiB of raw BF16. Squeezing the DiT further is not the lever — it does not
even break at the smallest legal groupsize (§4). The encoder is.

Until 2026-09-12 quantizing it bought memory and nothing else: stock ComfyUI runs every text encoder
with dequantized math behind two independent locks, so the 4-bit kernel was never reached. Both
locks now come off for a checkpoint that really carries quantized layers, and
`--disable-quantized-text-encoder` puts the old path back. See
[`cadeado_text_encoder.md`](cadeado_text_encoder.md).

The `krea2` encoder needed a new converter profile: `qwen3vl_4b` puts its decoder one segment
deeper than plain Qwen — `model.language_model.layers.N`, because a vision tower shares the file —
so the existing `qwen` profile matched **zero** layers. The `qwen3vl` profile matches 252, exactly
36 layers × 7 Linears, with zero tensors from the vision tower.

### Both formats, both prompt lengths, against the BF16 original

Median of 5 encodes per cell, each arm in its own child process. The reference is the BF16 encoder,
not the other arm — both quantized arms share the same weights, so "how much they differ from each
other" is not the question that decides.

| encoder | GiB | 76 tokens: rel-RMSE / ms | 1220 tokens: rel-RMSE / ms |
|---|---|---|---|
| `qwen3vl_4b_bf16` | 8.27 | — / 96.4 | — / ~702 |
| `qwen3vl_4b_w4a8` **released** | **3.41** | **0.1438** / **87.2** | **0.5839** / **258.0** |
| `qwen3vl_4b_w4a8` locked | 3.41 | 0.1438 / 182.3 | 0.5848 / 787.2 |
| `qwen3vl_4b_w4a4_convrot` released | 3.19 | 0.5016 / 77.2 | 0.7393 / 215.5 |
| `qwen3vl_4b_w4a4_convrot` locked | 3.19 | 0.3637 / 97.2 | 0.7017 / 712.2 |

**Take W4A8.** It is **2.53x more faithful than W4A4** at the short prompt (0.1438 against 0.3637)
for 0.22 GiB more, and releasing the lock costs it **nothing** measurable — 0.14381 against 0.14382
— while making it 2.09x faster. W4A4's release is a real trade: 1.26x faster for 1.38x worse.

**Two things here were measured wrong on the first pass and are corrected:**

1. *"Releasing the lock makes W4A4 slower"* came from a single run per arm. At five repeats
   releasing is 1.26x **faster**. The first encode of any arm is 2.8–3.5x the median (warm-up), and
   with one measurement that noise is the result.
2. The probe timed the BF16 reference **once** while timing the arms N times. The same file, same
   prompt, measured 282.2 ms in one invocation and 434.1 ms in the next — 1.54x apart. A reference
   that varies more than the effect is not a reference. Fixed: the reference now repeats like the
   arms.

### What this costs in fidelity grows with the prompt, and that is the awkward part

W4A8's conditioning error goes **0.1438 at 76 tokens → 0.5839 at 1220**, cosine 0.9896 → 0.8456.
The speed win goes the same way (2.09x → 3.05x against the locked path, and 2.71x against BF16),
so the format is cheapest exactly where it is least accurate. There is no single number for "what
a quantized encoder costs" — it depends on how much text you send it.

**Caveat on that long-prompt column, stated because it weakens it:** the 1220-token prompt is
**synthetic and repetitive** — nine descriptive clauses repeated six times. Degenerate input may
stress a quantizer differently from natural long text, so the short-prompt column is the better
founded of the two.

### The control that had to pass

A non-quantized encoder must be untouched by the flag. `qwen3vl_4b_bf16` loaded in both arms:
**0 quantized layers, `force_cast_weights` True in both.** The float32 upcast stays exactly where
it was.

### The Krea 2 chain, end to end

| link | original | shipped quantized | factor |
|---|---|---|---|
| diffusion | 24.48 GiB | **7.50** GiB W4A4 | 3.26x |
| text encoder | 8.27 GiB | **3.41** GiB W4A8 | 2.43x |
| VAE | 0.24 GiB | not quantized, by decision | — |
| **total** | **33.0 GiB** | **11.2 GiB** | **2.95x** |

The VAE is deliberately out of scope: at 242 MiB against 7.5 GiB of DiT it cannot move the number,
and nothing on this bench has ever measured a quantized VAE.
