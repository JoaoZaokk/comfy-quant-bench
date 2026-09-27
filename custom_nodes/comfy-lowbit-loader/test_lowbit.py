"""Unit tests on synthetic tensors, run as a script from the portable root:

    python_embeded\\python.exe -s custom_nodes\\comfy-lowbit-loader\\test_lowbit.py

Not collected by pytest on purpose: pytest imports the package `__init__.py` first, and that imports
ComfyUI's model management, which initialises a CUDA context on a card this bench shares.
"""

import importlib.util
import pathlib
import sys
import types

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ComfyUI"))

from comfy.cli_args import args  # noqa: E402

# comfy.model_management probes the GPU at import time; the tests never need its device logic, and the
# one GPU test addresses CUDA directly.
args.cpu = True


def _load_package():
    # The folder name has a hyphen, so it is imported the way ComfyUI does: by path, as a package.
    pkg = types.ModuleType("lowbit_pkg")
    pkg.__path__ = [str(HERE)]
    sys.modules["lowbit_pkg"] = pkg
    mods = {}
    for name in ("kernel", "layout", "formats"):
        spec = importlib.util.spec_from_file_location(f"lowbit_pkg.{name}", HERE / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mods[name]
        spec.loader.exec_module(mods[name])
    return mods


def cases(values):
    def mark(fn):
        fn.cases = values
        return fn
    return mark


def needs_cuda(fn):
    fn.needs_cuda = True
    return fn


M = _load_package()
kernel, layout, formats = M["kernel"], M["layout"], M["formats"]


def _affine(n, k, bits, g, levels, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, levels, (n, k), generator=gen)
    scale = (torch.rand(n, k // g, generator=gen) * 0.02 + 0.001).to(dtype)
    zero = (-scale.float() * (levels - 1) / 2).to(dtype)
    w = kernel.dequantize_torch(kernel.pack_codes(codes, bits), scale, zero, bits, g, torch.float32)
    return codes, scale, zero, w


@cases([(1, 2), (2, 3), (2, 4), (4, 16)])
def test_pack_then_dequantize_matches_formula(bits, levels):
    codes, scale, zero, w = _affine(64, 512, bits, 128, levels)
    expected = codes.view(64, 4, 128).float() * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)
    assert torch.equal(w, expected.view(64, 512))


@cases([(1, 2), (2, 3)])
def test_dense_packing_is_lossless_and_picks_the_smallest_width(bits, levels):
    _, _, _, w = _affine(256, 1024, bits, 128, levels)
    w = w.to(torch.bfloat16)
    layer = formats.pack_dense_layer(w)
    assert layer is not None and layer.bits == bits and layer.group_size == 128
    back = kernel.dequantize_torch(layer.qdata, layer.scale, layer.zero, layer.bits, 128, torch.bfloat16)
    assert torch.equal(back, w)


def test_dense_packing_refuses_a_normal_weight():
    assert formats.pack_dense_layer(torch.randn(256, 1024, dtype=torch.bfloat16)) is None


def test_gemlite_storage_is_the_transposed_canonical_layout():
    codes, scale, zero, w = _affine(96, 384, 2, 128, 3, dtype=torch.float32)
    canonical = kernel.pack_codes(codes, 2)                      # (N, K/4)
    gem_wq, gem_s, gem_z = canonical.t().contiguous(), scale.t().contiguous(), zero.t().contiguous()
    back = kernel.dequantize_torch(gem_wq.t().contiguous(), gem_s.t().contiguous(), gem_z.t().contiguous(), 2, 128, torch.float32)
    assert torch.equal(back, w)


def test_mlx_uint32_words_are_the_canonical_bytes():
    codes, scale, zero, w = _affine(8, 256, 2, 128, 3)
    words = kernel.pack_codes(codes, 2).view(torch.int32)        # what mx.quantize stores (as uint32)
    assert torch.equal(words.view(torch.uint8), kernel.pack_codes(codes, 2))


def test_shape_params_recovers_bits_and_group():
    q = torch.zeros(10, 3072 // 4, dtype=torch.uint8)
    assert layout.shape_params(q, torch.zeros(10, 24), 3072) == (2, 128)
    try:
        layout.shape_params(q, torch.zeros(10, 24), 3000)
    except ValueError:
        return
    raise AssertionError("a K that does not fit the packed width was accepted")


def test_diffusers_qkv_fuse_concatenates_packed_rows():
    parts = {}
    for i, c in enumerate("qkv"):
        codes, scale, zero, _ = _affine(16, 256, 2, 128, 3, seed=i)
        parts[f"transformer_blocks.0.attn.to_{c}"] = formats.LowBit(kernel.pack_codes(codes, 2), scale, zero, 2, 128)
    dense = {"double_stream_modulation_img.linear.weight": torch.zeros(4, 4),
             "norm_out.linear.weight": torch.arange(8.0).view(8, 1)}
    out_dense, out_lowbit = formats.diffusers_flux2_to_bfl(dense, parts)
    fused = out_lowbit["double_blocks.0.img_attn.qkv"]
    assert fused.qdata.shape == (48, 64)
    assert torch.equal(fused.scale[16:32], parts["transformer_blocks.0.attn.to_k"].scale)
    assert torch.equal(out_dense["final_layer.adaLN_modulation.1.weight"].flatten(), torch.tensor([4., 5, 6, 7, 0, 1, 2, 3]))
    assert "double_stream_modulation_img.lin.weight" in out_dense


@needs_cuda
@cases([(1, 2), (2, 3)])
def test_triton_matches_torch_bit_for_bit(bits, levels):
    codes, scale, zero, _ = _affine(3072, 3072, bits, 128, levels)
    q, s, z = kernel.pack_codes(codes, bits).cuda(), scale.cuda(), zero.cuda()
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        assert torch.equal(kernel.dequantize_triton(q, s, z, bits, 128, dtype), kernel.dequantize_torch(q, s, z, bits, 128, dtype))


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        if getattr(fn, "needs_cuda", False) and (kernel.triton is None or not torch.cuda.is_available()):
            print(f"SKIP {name} (needs CUDA + Triton)")
            continue
        for case in getattr(fn, "cases", [()]):
            try:
                fn(*case)
                print(f"ok   {name}{case or ''}")
            except Exception as e:  # report every failure, then exit non-zero
                failed += 1
                print(f"FAIL {name}{case or ''}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
