# Regras rígidas: text encoders

> Referência preservada do CLAUDE.md original, linhas 554–609, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.


  **And for a text encoder the rule is unreachable by construction, whatever the checkpoint says.** Measured 2026-08-31 on the 3090 against a real 13.20 GiB public W4A4 encoder (`qwen3vl_32b_minimax_h3-int4_convrot`, 350 `convrot_w4a4` layers, `linear_dtype` absent so the signature default `"int4"` applies). Called by hand, the kernel is genuinely native — 15/15 outputs differ from the forced INT8 fallback. Loaded through the stock `CLIPLoader` path and *counted* during a real encode:

  ```
  100/100 Linear   MixedPrecisionOps.Linear, quant_format convrot_w4a4, weight QuantizedTensor
  _convrot_w4a4_forward        0
  QuantizedTensor.dequantize 350
  ```

  Zero. The weight stays 4-bit in VRAM and the **math is dequantized** — memory saved, no time saved, kernel never reached.

  **The obvious culprit is the wrong one, and flipping it changes nothing.** `comfy/sd1_clip.py:114` hardcodes `full_precision_mm=True` for every text encoder; setting it False left the count at 0. Instrumenting each term of `_use_quantized` (`comfy/ops.py:1372-1377`) on a real forward names the one that actually bites: **`comfy_force_cast_weights=True`**, which comes from `model_patcher.py:743` via `set_model_compute_dtype`, called at **`comfy/sd.py:269` for every CLIP object** — `set_model_compute_dtype(torch.float32)`, commented "Match torch.float32 hardcode upcast in TE implemention". Two independent locks, and only releasing both fires the kernel.

  **Release it at the source, never on the modules — and this cost a run that looked like a result.** `model_patcher.py:1016` rewrites `m.comfy_force_cast_weights = self.force_cast_weights` on every module *every time the model is loaded to GPU*. Writing the attribute on the modules survives only if the model happened to be resident already, which depends on VRAM state at that instant. **The same command line gave 350 kernel calls on one run and 0 on the next.** The fix is `clip.patcher.force_cast_weights = False` (plus popping `manual_cast_dtype`), which `patch_model` then propagates. What caught it was the dispatch counter reporting 0 — a probe that only compared outputs would have reported "releasing the locks changes nothing" and been believed.

  With that fixed, the same measurement on three real encoders, each `--sem-bf16` except Qwen, which has a BF16 twin on disk:

  ```
  encoder                 formato  quant   travado  destravado    tempo            erro C-vs-B   cos
  qwen_3_4b (4B)            W4A4  2.4 GiB   70.2ms      82.9ms   1.18x MAIS LENTO    5.99e-1   0.949
  gemma_3_12B_heretic       W4A8  8.1 GiB  1772.3ms    478.9ms   3.70x mais rapido   2.11e-1   0.982
  qwen3vl_32b_minimax       W4A4 13.2 GiB   311.6ms    117.7ms   2.65x mais rapido   9.73e-2   0.99989
  ```

  **The sign is set by the sequence length, and that took varying one axis alone.** The first version of this paragraph asserted a mechanism it had not measured ("the cost scales with weight size while the token count stays tiny"), and the table above refuses it: the locked times are 70.2 ms at 2.4 GiB, **1772.3 ms at 8.1 GiB**, 311.6 ms at 13.2 GiB — not monotonic in weight size. The three encoders differ in weight size *and* in sequence length at once (Gemma's conditioning is `[1, 49, 1024, 3840]`, MiniMax's `[1, 8, 5120]`), so that table cannot separate them. Same file, same card, only the prompt changed:

  ```
  qwen_3_4b W4A4, RTX 3080 Ti, mediana de 3
  tokens   travado  destravado
      22     80.7      120.1    1.49x MAIS LENTO
      75    100.1      105.8    1.06x MAIS LENTO
     199    151.9       95.2    1.60x mais rapido    <- cruza entre 75 e 199
     424    245.2      102.0    2.40x
     850    456.1      124.3    3.67x
    1496    824.5      249.0    3.31x
  ```

  **The crossover is between 75 and 199 tokens** on this model and card. The released path is nearly flat from 22 to 424 tokens (120 → 102 ms) while the locked path climbs with the sequence, so the 4-bit kernel carries a per-layer fixed cost that a short prompt cannot amortise and the dequantized path pays a BF16 GEMM that grows with the tokens. Real prompts usually sit above that crossover: at 850 tokens the quantized encoder also beats the **BF16 original** (372.1 ms at ~700 tokens against 124.3 ms). Weight size may still matter on top of this — MiniMax won at 8 tokens — but it has not been isolated. And **the accuracy cost is not a property of the format**: two W4A4 files differ by 6x in the error releasing adds (5.99e-1 against 9.73e-2), so "W4A4 costs X" cannot be quoted without naming the checkpoint; on Qwen the long prompt is also worse than the short one (weight-only 2.55e-1 against 1.44e-1). For Qwen the full picture is available: the 4-bit *weight* alone already costs 1.44e-1 against its BF16 twin, and releasing takes it to 6.09e-1 — 4.23x.

  **Measured the same day on this project's own outputs, and the split is clean.** `tools/probe_quant_dispatch.py` counts the same way against a real load and a real forward:

  ```
  zimage-v2-w4a4        difusao  170 convrot_w4a4   340 quantizados, 0 sem, 0 dequantize
  gemma_3_12B_heretic   TE       336 asym_w4a8_int8   0 quantizados, 336 sem, 336 dequantize
  ```

  So: **the diffusion path really does run quantized** — every Z-Image number this bench has published (per-layer error, epsilon per step, the INT4-vs-INT8 comparison) was measured on genuinely quantized math, not on two dequantizations. And **our own Gemma is memory-only**, exactly as predicted, with both locks `True` on all 336 layers. The format does not matter: `asym_w4a8_int8` is blocked by the same `comfy_force_cast_weights`, so this is about being a text encoder, not about being W4A4.

  One counting trap worth keeping: `MixedPrecisionOps.Linear` is the class of **every** Linear in the model, quantized or not. Patching the class and counting all its calls first reported Z-Image as "MISTO, 340 against 76" — the 76 were layers that never had a quantized weight at all. Scope the count to `layout_type is not None`.

  **Still not measured:** what it costs in s/it on a real LTX render, and whether releasing the locks on a text encoder is safe for output quality.

  **The MiniMax half of that caveat is closed since 2026-08-31: its BF16 twin is now on disk.** `Comfy-Org/MiniMax-H3`, `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`, 47.97 GiB, byte-exact against the published size, **351 of 351 layer names matching** the Winnougan quantization. Measured with `tools/probe_winnougan_fidelidade.py` over 20 layers, 5 shapes, 4 depths (0/16/33/49), M in {1, 64, 1024}: **int8 1.40x more faithful, 60 of 60**. That makes three independent measurements in the same direction — 1.49x on Z-Image with real activations, 1.33x on epsilon-per-step, 1.40x here — across a different model family, a **different quantizer** (theirs, not ours), and a different activation kind. Per-layer error is also flat in depth: same shape, block 0 to block 49, 2.2776e-1 → 2.2762e-1.

  Not covered: no SASS; there is no BF16 reference for the Gemma on this bench, so its error column is against its own dequantized arm and is **not** a fidelity claim; one prompt per encoder, one card; the locks were released by post-load monkeypatch, not by anything ComfyUI offers — `custom_operations` in `model_options` is the only real escape and no node exposes it. Re-run with `tools/probe_winnougan_int4.py`, `tools/probe_winnougan_load.py`, `tools/probe_te_fullprecision_mm.py` and `tools/probe_te_lock_cost.py`.

