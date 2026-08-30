# Mixed-Precision Quantization Criteria: LTX-2.5 vs Krea-2

## LTX-2.5-Quantized (joeygambino)

**Criterion**: Greedy error allocation, budget-constrained to target size. Ranking metric is "error-removed-per-byte" **weighted by layer magnitude (||W||²)**, not relative error alone.

**Why**: Relative 4-bit error spans only 2% across layers (0.0721–0.0737), useless for ranking. Weight magnitude clustering emerges only after: "363 of first 386 promotions land in audio tower, only 23 in video tower" — audio-critical components prioritized.

**Format**: `comfy-mix4x8-*.safetensors` with per-layer `.comfy_quant` metadata (JSON in uint8 tensor) + `__metadata__["_quantization_metadata"]["layers"]`.

## Krea-2-SVDQuant (AlperKTS)

**Criterion**: Architectural sensitivity, not dynamic measurement. Only 224 transformer-block linears quantized; norms, modulation, text-fusion, final layer stay FP32 (small + disproportionately sensitive).

**Activation stats role**: `--act-stats` weights the minimization objective by measured per-channel activation RMS instead of uniform weighting. Verified: more sophisticated `--rank-alloc` strategies did NOT improve image quality.

**Format**: Identical to LTX-2.5 — `.comfy_quant` metadata tensors storing format per layer.

## Format Compatibility

Both use ComfyUI's native `.comfy_quant` marker (JSON-encoded config in uint8 tensor). ComfyUI 0.2.26+ reads `__metadata__["_quantization_metadata"]["layers"]` as authoritative source of truth. **Format is compatible** — metadata travels in safetensors headers, per-layer precision tags are identical structure.

## Prior Art (Arxiv 2024–2025)

- **MPQ-DM/MPQ-Diff**: Cross-layer correlation ("network orthogonality") as sensitivity proxy per timestep
- **DiffPro**: 6-stage pipeline: sensitivity → per-layer refinement → dynamic activation → pruning → budgeted search → deploy
- **ViDiT-Q**: Timestep + layer-wise precision search
- **MXSens**: Sensitivity-aware framework
- **BRECQ**: Blockwise reconstruction (LLM-origin, adapted for diffusion)

All measure error or sensitivity; none use pure architecture-driven heuristics like Krea-2's approach.

---

**Conclusion**: LTX-2.5 measures reconstruction error (weighted), Krea-2 uses architectural rules (unweighted). Both discovered that dynamic per-layer allocation beats uniform precision on quality vs size tradeoff. **Format is fully compatible** — the `comfy_quant` metadata standard is designed exactly for this mixed-precision interchange.

Sources:
- [LTX-2.5-Quantized README](https://huggingface.co/joeygambino/LTX-2.5-Quantized)
- [Krea-2-SVDQuant-ComfyUI quantize_krea2.py](https://github.com/alperktt/Krea-2-SVDQuant-ComfyUI)
- [ComfyUI QUANTIZATION.md](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md)
- [DiffPro arxiv:2511.11446](https://arxiv.org/pdf/2511.11446)
- [MPQ-Diff arxiv:2412.00144](https://arxiv.org/abs/2412.00144)
- [ViDiT-Q arxiv:2406.02540](https://arxiv.org/pdf/2406.02540)
- [MXSens arxiv:2607.17733](https://arxiv.org/html/2607.17733)
