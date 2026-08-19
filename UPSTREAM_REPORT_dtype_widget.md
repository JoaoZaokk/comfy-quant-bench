# Draft upstream report: `weight_dtype` widget is obeyed on checkpoints that carry `quant_config`

**Status: written, NOT published.** No remote is configured on this repository, and opening an
issue or a PR against ComfyUI is an action directed outside this machine. It waits for the user.

ComfyUI `v0.33.0-19-gc1739380`, Windows 11, RTX 3090, torch 2.13.0+cu130, comfy-kitchen 0.2.23.

---

## Summary

`load_diffusion_model_state_dict` checks `model_config.quant_config is not None` twice and uses it
to keep dtype machinery away from a checkpoint that carries its own per-layer quantization -- once
for the sniffed `weight_dtype`, once for `manual_cast_dtype`. The explicit `dtype` from the
`UNETLoader` widget is not covered by either check, so it becomes `unet_dtype` unconditionally.

The result is not a crash. The quantized layers are unaffected -- they are already 4-bit and keep
dispatching to their kernels -- but every tensor the quantization profile deliberately left in high
precision is cast to fp8, and the output changes, with no error, no warning, and no slowdown.

## Code

`comfy/sd.py`, current `master` at time of writing:

```python
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None                                    # guarded

    if dtype is None:
        unet_dtype = model_management.unet_dtype(model_params=parameters,
                                                 supported_dtypes=unet_weight_dtype,
                                                 weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype                                     # NOT guarded

    if model_config.quant_config is not None:
        manual_cast_dtype = model_management.unet_manual_cast(
            None, load_device, model_config.supported_inference_dtypes)   # guarded
    else:
        manual_cast_dtype = model_management.unet_manual_cast(
            unet_dtype, load_device, model_config.supported_inference_dtypes)
```

The same shape appears in the sibling path, where it reads
`unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))` before the
`quant_config` test, so a fix would touch both.

## Reproduction

Any checkpoint carrying `_quantization_metadata`, loaded through `UNETLoader` with `weight_dtype`
set to `fp8_e4m3fn` or `fp8_e5m2`. Measured here on a Z-Image checkpoint with 170 quantized layers
(115 `convrot_w4a4`, 55 `asym_w4a8_int8`), 4 steps, 512px, identical seed and prompt, changing only
the widget:

| widget | `unet_dtype` | unquantized tensors after load | latent norm |
|---|---|---|---|
| `default` | `bfloat16` | `bf16` x283 | 747.06 |
| `fp8_e4m3fn` | `float8_e4m3fn` | `bf16` x76, `float8_e4m3fn` x207 | 713.98 |
| `fp8_e5m2` | `float8_e5m2` | `bf16` x76, `float8_e5m2` x207 | 828.08 |

`manual_cast_dtype` stays `bfloat16` in all three, so the guard on that line does its job. All
three ran 680 of 680 quantized dispatches on the native kernels, counted by wrapping the layout
dispatch table -- the widget cannot reach the quantized weights.

For scale: applying a LoRA at strength 1.0 to the same model with the same seed moves the latent
norm from 747.06 to 728.99. The `fp8_e5m2` widget moves it to 828.08, about four times further.

## What is not claimed

* Whether this is intended is not established. `git log -S` on this checkout shows the whole block
  arriving in one templates-update commit, which says nothing about intent for these three lines.
  It may be deliberate that an explicit user dtype wins; the report is that it wins *silently*, and
  that the two adjacent guards suggest otherwise.
* No image-quality evaluation was done, only latent statistics. "The output changed by this much"
  is measured; "the output got worse" is not.
* Only two quantization formats were exercised.

## Possible fixes, in increasing order of assumption

1. Log at `INFO` when an explicit `dtype` is applied to a model with `quant_config` set, naming how
   many tensors it will affect. Costs nothing and removes the silence, which is the actual harm.
2. Make the explicit path consistent with its two neighbours -- ignore the widget when
   `quant_config` is set.
3. Surface it in the UI so `UNETLoader` greys the widget for a quantized file.

The first is the one this report would propose, because it does not require guessing intent.
