# NVFP4 Support on Ampere (sm86): Research Findings

**Data**: 2026-08-30  
**GPU Target**: RTX 3090 + RTX 3080 Ti (both Ampere, sm86)  
**Question**: Can NVFP4 checkpoints run on Ampere?

## Primary Question: NVFP4 on Ampere sm86

### **ANSWER: (a) Does NOT run**

**Evidence Level**: CONFERIDO (verified via official sources + community confirmation)

#### VISTO (Sources State):
- NVIDIA official: NVFP4 is **exclusive to Blackwell** architecture (SM100 min, SM120 desktop)
- RTX 3090/3080 Ti are Ampere (sm86) — **compute capability 86**
- Blackwell requires **compute capability 100+**
- Zero mention of emulation or fallback path

#### CONFERIDO (Verified):
- X/Twitter (Grok): *"No, it's a no-go on the 3090. The NVIDIA NVFP4 build ... is specifically for Hopper and Blackwell GPUs"*
- No native FP4 tensor cores on Ampere
- No software emulation route documented
- GPTQ/INT4/Q4_K_M recommended as alternatives

#### Affected Models (inapplicable here):
- ultranationalism/Z-Image-Turbo-SVDQuant-NVFP4 (ModelScope) — **cannot run on 3090/3080Ti**
- ModelsLab/MiniMax-H3-svdquant-nvfp4_r32 (ModelScope) — **cannot run**
- joeygambino/LTX-2.5-Quantized (LTX25-distilled-DiT-comfy-nvfp4.safetensors) — **cannot run**

---

## Secondary Finding: OrbitQuant

### What is OrbitQuant?

**Status**: Separate method, NOT ConvRot with another name

**Core Difference**:
- **ConvRot** (W4A4): Conv-Rot kernel, hardware-native on CUDA
- **OrbitQuant** (W4A4): Post-training quantization using randomized permuted block-Hadamard (RPBH) rotation + Lloyd-Max codebooks, **data-agnostic** (no calibration data needed)

**Key Characteristics**:
- Quantizes both weights AND activations to 4-bit (W4A4)
- Works on vision transformers/diffusion models (FLUX.1, Wan 2.1/2.2)
- Requires **custom OrbitQuant runtime** (not standard CUDA)
- Paper: arxiv.org/abs/2607.02461

**Compute Capability**: Not specified in available documentation; requires consulting ApacheOne/orbitquant_unets repository

---

## Actionable Conclusion

The three NVFP4 checkpoints in round 1 are **inapplicable** to RTX 3090/3080 Ti. Research direction should pivot to:

1. **ConvRot W4A4** (native, already benchmarked)
2. **SVDQuant** (Nunchaku, supports sm80/sm86 natively)
3. **GPTQ/INT4** (proven on Ampere)

NVFP4 requires Blackwell (RTX 5090+, H200, etc.)

---

## Sources

- [NVIDIA Blog: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA Blackwell Ultra Deep Dive](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [Spheron: NVFP4 vs MXFP4 GPU Requirements](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/)
- [X/Twitter Grok Confirmation](https://x.com/grok/status/2072194183767429187)
- [OrbitQuant Paper](https://arxiv.org/abs/2607.02461)
- [Nunchaku Hardware Compatibility](https://deepwiki.com/mit-han-lab/nunchaku/1.2-hardware-compatibility)
