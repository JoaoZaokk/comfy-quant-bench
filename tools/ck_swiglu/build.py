"""Build the comfy-kitchen SwiGLU+ConvRot quantizer companion module (`_swiglu_quant`) and optionally install it.

The kernel (swiglu_quant.cu) includes comfy-kitchen's own ops/convrot_w4a4.cu, so it needs a comfy-kitchen source
checkout whose quantizer matches the installed wheel. The check below compares the include files (line endings
normalized) with the v0.2.35 blobs; the installed comfy-kitchen here is 0.2.35 and v0.2.37 has identical files.

    tools/ck_swiglu/build.bat [--ck-src DIR] [--install]

--install copies build/_swiglu_quant.pyd into site-packages/comfy_kitchen/backends/cuda/, where the patched
backends/cuda/__init__.py (patches/comfy_kitchen_swiglu_w4a4_fused.patch) imports it. Without the module the patch
falls back to torch's silu(gate) * up and the regular quantizer, i.e. the unpatched behaviour.
The module is a regular torch extension: rebuild it after any torch upgrade (an import failure also falls back).
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CK_SRC = ROOT / ".scratch" / "quantfunc_2026-10-01" / "fase3" / "comfy-kitchen"
# sha256 of the v0.2.35 blobs (LF line endings) of every file the kernel pulls in
EXPECTED = {
    "ops/convrot_w4a4.cu": "051d2fb58e848398",
    "ops/svdquant_utils.cuh": "de71c9b49b141279",
    "dtype_dispatch.cuh": "5a155c3bf2123604",
    "float_utils.cuh": "5dd76a0abac5c556",
    "utils.cuh": "a3482d9911029f01",
}


def check_sources(cuda_dir):
    bad = []
    for rel, want in EXPECTED.items():
        p = cuda_dir / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
            continue
        got = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
        if got != want:
            bad.append(f"{rel}: {got} != {want}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck-src", type=Path, default=DEFAULT_CK_SRC, help="comfy-kitchen source checkout")
    ap.add_argument("--install", action="store_true", help="copy the module into the installed comfy_kitchen")
    ap.add_argument("--any-ck", action="store_true", help="skip the source-hash check (not recommended)")
    args = ap.parse_args()

    cuda_dir = args.ck_src / "comfy_kitchen" / "backends" / "cuda"
    bad = check_sources(cuda_dir)
    if bad and not args.any_ck:
        sys.exit("comfy-kitchen source does not match the installed quantizer:\n  " + "\n  ".join(bad))

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    import torch
    from torch.utils.cpp_extension import load

    bd = HERE / "build"
    bd.mkdir(exist_ok=True)
    t0 = time.time()
    mod = load(
        name="_swiglu_quant",
        sources=[str(HERE / "swiglu_quant.cu")],
        extra_include_paths=[str(cuda_dir), str(cuda_dir / "ops")],
        extra_cflags=["/O2", "/Zc:preprocessor"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor", "--expt-relaxed-constexpr",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-gencode", "arch=compute_80,code=sm_80",
                           "-gencode", "arch=compute_86,code=sm_86", "-gencode", "arch=compute_86,code=compute_86"],
        build_directory=str(bd),
        verbose=False,
    )
    pyd = Path(mod.__file__)
    digest = hashlib.sha256(pyd.read_bytes()).hexdigest()
    print(f"built {pyd} in {time.time() - t0:.0f} s (torch {torch.__version__}, sha256 {digest[:16]})")
    if args.install:
        import comfy_kitchen.backends.cuda as ckc
        dst = Path(ckc.__file__).parent / "_swiglu_quant.pyd"
        shutil.copy2(pyd, dst)
        print(f"installed {dst}")


if __name__ == "__main__":
    main()
