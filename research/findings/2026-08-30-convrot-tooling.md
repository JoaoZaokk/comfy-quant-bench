# ConvRot W4A4 Quantization Tooling Discovery
**Date**: 2026-08-30  
**Scope**: Locate the tool that generates checkpoints with `convrot_w4a4_mixed`, `calibrated_plus_preset`, `BalancedQ` fields  
**Control**: Strings searched LITERAL across GitHub, HuggingFace, ModelScope, Gitee; absence recorded per source

## What was found

### Paper & Official Repository
- **arXiv:2512.03673**: "ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers" (Huang et al.)
- **Official code**: https://github.com/feice-huang/ConvRot
  - Inference + offline quantization entry points
  - CUDA butterfly-rotation kernel for group-wise Hadamard rotation
  - Per-model YAML configs for FLUX, SD3, DiT
  - Benchmarks showing 4× memory savings, 2× speedup on diffusion transformers
  - **Does NOT generate the metadata fields in question** (verified against README)

### Quantized Checkpoints
- **Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot** (HuggingFace)
  - 67114 downloads on ModelScope (as stated in task)
  - Mixed INT4/INT8 variants for Minimax H3
  - **README contains no quantization tool attribution or link** (verified via fetch)

### ComfyUI Quantization Ecosystem
- **NidAll/comfyui-mixed-quantizer**: per-layer W4A4/W4A8/INT8/BF16 selection
  - Presets: `balanced`, `conservative`, `size-first` (NOT `BalancedQ`)
  - Does not generate `convrot_w4a4_mixed` or `calibrated_plus_preset` fields
- **SparknightLLC/ComfyUI-QuantizationToolkit**: W4A4/W4A8/W8A8 paths
  - No mention of `BalancedQ` or `calibrated+`
- **Comfy-Org/comfy-quants**: utility scripts (FP8, INT8, NVFP4, AWQ)
  - Does not generate the specific metadata fields

## What was NOT found

### Literal strings searched (no results in public repos)
| String | Search scope | Result |
|--------|------------|--------|
| `calibrated+` | GitHub, HuggingFace, arXiv | NOT FOUND |
| `BalancedQ` | GitHub, HuggingFace, arXiv | NOT FOUND |
| `convrot_w4a4_mixed` | Safetensors metadata in public repos | NOT FOUND (only `convrot_w4a4`) |
| `int8_mm_ratio` | GitHub code search, docs | NOT FOUND |
| `w4a4_int4mm_layers` | GitHub code search, docs | NOT FOUND |
| `calibrated_plus_preset` | GitHub, HuggingFace | NOT FOUND |
| `prune_donor` | GitHub safetensors context | NOT FOUND in convrot context |

### Absence control (what was checked to confirm)
- GitHub `site:github.com` + all 6 field names: 0 hits
- HuggingFace `site:huggingface.co` + `convrot_w4a4_mixed` OR `calibrated_plus_preset` OR `BalancedQ`: found ConvRot repos but NOT those fields
- ModelScope `site:modelscope.cn` + convrot metadata: no specific results for these fields
- Gitee search: no Abiray presence, no relevant repos

## Conclusion

**The tool that generates checkpoints with these exact metadata fields does not appear in public repositories.** Possibilities:

1. **Private repository** hosted on Alibaba ModelScope / internal infrastructure
2. **Unpublished proprietary tool** used internally for Minimax quantization
3. **Fork/variant** of ConvRot with custom metadata schema not integrated upstream
4. **Tool exists but named differently** — strings may be internal implementation details not documented

The official ConvRot paper and code exist and are well-documented. The checkpoints exist (Abiray's Minimax collection). **The quantization workflow that produces these specific metadata fields is not publicly linked in any discoverable source.**

## Next steps for the bench

- Check if Abiray has a GitHub account with quantization code
- Look for ModelScope-only repositories (require WeChat/Chinese registration)
- Search Alibaba's internal/research quantization tooling documentation
- Ask Minimax community directly in discussions

**Sources**:
- [ConvRot official paper arXiv:2512.03673](https://arxiv.org/abs/2512.03673)
- [feice-huang/ConvRot GitHub](https://github.com/feice-huang/ConvRot)
- [Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot HuggingFace](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot)
- [NidAll/comfyui-mixed-quantizer GitHub](https://github.com/NidAll/comfyui-mixed-quantizer)
- [Comfy-Org/comfy-quants GitHub](https://github.com/Comfy-Org/comfy-quants)
