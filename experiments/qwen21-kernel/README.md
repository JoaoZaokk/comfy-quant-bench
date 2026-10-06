# qwen21-kernel: phase-7 experiment code

Reference code behind [docs/kernel-optimization.md](../../docs/kernel-optimization.md): an own ConvRot W4A4 GEMM with
per-group scales for sm86, the activation quantizer, the test node that loads them into ComfyUI, and the two
step-switch workflows. Research code from one bench (RTX 3090, Windows, ComfyUI 0.37.x, Torch 2.13 + cu130), kept for
reproduction, not a supported package.

| path | what |
|---|---|
| `kernels/g64_gemm3.cu`, `kernels/g64_gemm6.cu` | IMMA m16n8k64 s4 GEMM; v3 = policies 0–3, v6 = row/row, g128/g128 FP and integer-ratio policies (0, 4, 5, 7, 15, 17), five tile configs |
| `kernels/g64_quant.cu`, `kernels/g64_quant2.cu` | Hadamard-256 rotation + per-group INT4 activation quantizer, optional fused SwiGLU; v2 adds a clip ratio |
| `kernels/build_g64.py` | builds one `.cu` with `torch.utils.cpp_extension.load` into `kernels/build_g64/<name>/` |
| `kernels/g64_quant_test.py`, `kernels/g64_quant2_test.py` | bit-exact correctness against a PyTorch reference, and interleaved timing |
| `kernels/g64_bench6.py`, `kernels/ubench_imma.*` | GEMM bench against the ConvRot CUTLASS kernel; IMMA micro-benchmark |
| `node/__init__.py` | test node: rotates and quantizes the 192 Qwen-Image-2.1 linears at load and routes them through the kernels above (`G64_POL`, `G64_MOD`, `G64_CFG`, `G64_SUBST`, `G64_TROCA`; see its docstring) |
| `workflows/QWEN21-TXT2IMG-int8-w4a4-k{3,5}.json` | INT8 ConvRot for the first 3 / 5 steps, then W4A4, with two stock `KSamplerAdvanced` nodes |
| `analysis/estat_bateria.py` | paired cluster bootstrap, t-test and Wilcoxon over battery results |
| `analysis/attn_int_emul.py` | INT4 / FP4 / INT8 QK and PV error emulated on a dumped attention call |

Benches take the GPU lock through `tools/_bench_guard.py`. The workflows need no custom node, only the INT8 ConvRot
(`tools/quant_int8.py --convrot`) and W4A4 ConvRot (`tools/quant_w4a4.py`) checkpoints of Qwen-Image-2.1.
