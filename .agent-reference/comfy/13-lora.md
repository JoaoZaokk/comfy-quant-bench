# LoRA sobre pesos quantizados

> Referência preservada do CLAUDE.md original, linhas 1232–1322, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## LoRA over a quantized weight: it is a requantization, and that is measured

The owner asked on 2026-09-13 whether LoRAs "work the way they should" on the quantized builds. The
only prior measurement here (2026-08-19) answered a narrower question — the native kernel still
fires with a LoRA applied, 680/680. **Whether the LoRA arrives intact in the weight had never been
measured.** Traced first, then measured on the real path (`tools/probe_lora_requant.py`):

`LoraLoaderModelOnly` over a `QuantizedTensor` does not keep the LoRA as a separate branch.
`ModelPatcher.patch_weight_to_device` (`comfy/model_patcher.py:899`) runs `convert_weight` →
`W.dequantize()` (`comfy/ops.py:1449`), adds the delta in `lora_compute_dtype` (fp16 on this
card), then `set_weight` → `W.requantize_from_float(W', scale="recalculate",
stochastic_rounding=seed)` (`comfy/ops.py:1455-1457`). **Dequantize, add, requantize to the same
4 bits.** The kernel is unchanged because the weight comes back in the same layout; the delta, which
is usually smaller than the 4-bit grid step, only survives in expectation.

Measured per layer on the real path, `err` = `‖W − W_bf16‖/‖W_bf16‖` (weight-space, **not** the
activation-space `err_w4a4` of the band table above — the two do not compare):

```
model, format              LoRA                       |δ|/|W|   survival  cosine  noise/LoRA   err: before -> no-op requant -> with LoRA
Z-Image v2  W4A4 cg256     RealisticSnapshot r32       0.095     1.000     0.51     1.7x        0.157 -> 0.163 -> 0.238   (1.45x)
Krea2 Turbo W4A4 cg256     krea2 turbo LoRA r64        0.0086    1.000     0.14     7.2x        0.162 -> 0.168 -> 0.176   (1.07x)
Wan 2.2 5B  W4A8           a 14B LoRA (wrong model)    0 (shape fails)  -   -       -           0.0731 -> 0.0835          (1.14x, NO LoRA applied)
LTX 2.5 22B W4A8           ltx2-squish (an LTX 2.0 LoRA) 0.013-0.086, ZERO in 16/24  0.956  0.39   3.1x    0.0731 -> 0.0837 -> 0.0837 (zero) .. 0.114
LTX 2.5 22B W4A8           LTX23 Product Commercial r16 0.0021    0.909     0.05    19x         0.0731 -> 0.0836 -> 0.0838   (1.15x)
LTX 2.3 22B W4A8           LTX23 Product Commercial r16 0.0020    0.908     0.05     20x        0.0731 -> 0.0837 -> 0.0838   (1.15x)
Qwen-Edit 2511 W4A8        Lightning 4-step r64         0.0005    0.858     0.01    103x        0.0731 -> 0.0836 -> 0.0835   (1.14x)
```

**And the output contradicts the weight, which is the finding.** Rendered the same day
(`bench/qwen_edit_lora/grade_lightning_4passos.png`, criterion R1-R4 written first in
`bench/criterio_lora.md`): at 4 steps without the LoRA both INT8 and W4A8 **fail** — no scarf, the
sign stays OPEN, a speckled apple-pear hybrid, oversharpened texture (the control that had to
fail, failed); with the LoRA, INT8, W4A8 merged and W4A8 bypass **all obey all three instructions**;
merged vs bypass differ by 1.26 in the untouched region against 3.5-3.9 between either and INT8.
The layer where the LoRA looked most buried (86% survival, noise 103x the delta) is the one whose
output is intact. **Weight-space per-layer numbers rank and alarm; they do not decide** — the same
lesson this file already records for per-layer error across formats. Caveat: a 4-step LoRA changes
the whole regime and is the most robust kind; a subtle style LoRA was not tested at the output.

On LTX 2.5 W4A8 with `ltx2-squish` (49 frames, same seed, no trigger word in the prompt — a design
hole, so this measures the cost of loading the LoRA, not its effect): merged and bypass land on the
**same** composition and sit 6.3 MAE apart, against 15-18 from the no-LoRA reference and 28 from a
seed change; neither moved the audio level or timing, while the other seed moved both (RMS -28 vs
-20 dBFS, lag +78 ms). The requantization noise perturbs the output by a third of what the LoRA does
and a quarter of what a seed does. `bench/ltx25/lora/`.

**LTX 2.3 W4A8 with its own LoRA (`LTX23_Product_Commercial`, trigger `srx_commercial`), measured
at the output on 2026-09-14 on valid conditioning, two rounds of 49 frames.** With the trigger,
merged and bypass both execute the prompt's "rotating slowly" (motion 4.6 against the no-LoRA
reference's 1.25) and draw a different headphone: 41 MAE from the same-seed reference, against 57
for a seed change — the LoRA still moves less than a seed (R7 refuted, as on 2.5), and R9 was
undecidable because a commercial prompt already renders as a commercial without the LoRA. Merged
and bypass sit 7.5 apart (5.1 without the trigger) against 23–41 from the reference — a fifth of
the LoRA's effect, the third round with that proportion (R8 confirmed). Bypass lowered the audio
level both times (−2.2 / −4.7 dB; merged −0.1 / −2.6) and costs 24 % per step (0.68 against 0.55
s/it, from the sampler's progress bar); merged costs nothing. The same LoRA was not rendered on the
BF16 original, so "same effect as unquantized" is not measured. `bench/criterio_lora.md`,
`bench/ltx23/lora*/`.

The two LTX rows add two things. `ltx2-squish` ships **all-zero `lora_B` for 768 of its 1152
matrices** (every audio and cross-modal attention family, read from the file); ComfyUI matches the
key, applies a zero delta and requantizes the layer anyway, so two thirds of the layers that LoRA
names pay the +14% for nothing. And on the `asym_w4a8_int8` codebook layout the survival is **not**
1.000 — 0.956 and 0.909 — a small delta loses 5-9% in the requantization, a bias the convrot W4A4
rows do not show; with |δ|/|W| = 0.002 the added noise is 19x the LoRA itself.

Three things that hold across the rows. **The LoRA is there, in expectation** — survival 1.000 to
three decimals, stochastic rounding has no bias. **What lands in the weight is the delta plus noise
larger than the delta** — cosine 0.51 and 0.14; the noise grows with the LoRA's own magnitude
(each entry moves with probability ∝ |δ|/step), which is why the √2 "independent noise" prediction
written before the run held on Z-Image (1.45x) and was refuted on Krea2 (1.07x). **Requantization
is not idempotent**: a no-op patch already costs 0.157 → 0.163 (convrot) and 0.0731 → 0.0835
(asym W4A8 with codebook).

The Wan row is the trap that only exists on quantized models. A LoRA from another architecture
matched 300 keys **by name**, failed every `calculate_weight` on shape, and ComfyUI logged
`ERROR lora ... shape` and **continued** — delta zero, weight requantized anyway, model 14% worse in
weight error with nothing applied. On a BF16 model the same failure rewrites the weight unchanged
and is harmless. The only evidence is one log line per layer.

`LoraLoaderBypassModelOnly` (`comfy_extras/nodes_lora_debug.py`, labelled "for debugging") keeps the
delta as a BF16 low-rank branch in the forward and never touches the weight, so it has none of this
noise. Whether the noise is *visible* is a render question; the criterion for those renders was
written before running them in `bench/criterio_lora.md` (R1–R6), with the control that has to fail
— Qwen-Image-Edit at 4 steps **without** its Lightning LoRA — named first.

Not covered: one strength (1.0), one LoRA per model, a sample of layers; nothing on text-encoder
LoRAs; the bypass loader is measured only at the output, since by construction it leaves the
weight alone.

