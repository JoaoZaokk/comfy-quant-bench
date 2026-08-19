"""Assert where detect_model_route sends every architecture ComfyUI ships with double+single blocks.

Routing is the one decision that cannot be checked by running a model: picking the wrong branch
either raises somewhere deep inside the transformer, or -- worse -- runs and quietly does nothing.
So it gets asserted directly, against stand-ins built from the *real* comfy.ops Linear classes
rather than torch.nn.Linear, because that difference is exactly what broke an earlier version of
the gate: the Linear an fp8 checkpoint gets does not inherit from torch.nn.Linear at all.

    python tools/fbcache_routing_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402


def load_first_block_cache():
    import importlib.util

    node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "Comfy-WaveSpeed-Fixed"
    spec = importlib.util.spec_from_file_location(
        "wavespeed_fbc", node_dir / "first_block_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenRefiner(torch.nn.Module):
    """HunyuanVideo's text projection: three required arguments, not one."""

    def forward(self, x, c, mask, transformer_options={}):
        return x


class AVBlock(torch.nn.Module):
    """LTX-2.5's block shape: hidden state is (video, audio), and there are two text streams."""

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None,
                transformer_options=None):
        return x


class ZImageBlock(torch.nn.Module):
    """Z-Image's block: one sequence carrying caption and image tokens, no text stream at all."""

    def forward(self, x: torch.Tensor, x_mask=None, freqs_cis=None, adaln_input=None,
                transformer_options={}):
        return x


class PlainBlock(torch.nn.Module):
    """An unannotated block, which must not be refused on suspicion."""

    def forward(self, x, context=None, transformer_options={}):
        return x


class Params:
    def __init__(self, **kwargs):
        self.global_modulation = False
        self.vec_in_dim = None
        self.guidance_embed = False
        for key, value in kwargs.items():
            setattr(self, key, value)


def make(class_name: str, **attributes) -> torch.nn.Module:
    """A stand-in whose class name and attributes are what detect_model_route dispatches on."""
    cls = type(class_name, (torch.nn.Module,), {})
    module = cls()
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def make_subclass(base_name: str, class_name: str, **attributes) -> torch.nn.Module:
    """A stand-in whose *ancestor* carries the recognised name, as LTXAVModel(LTXVModel) does."""
    base = type(base_name, (torch.nn.Module,), {})
    module = type(class_name, (base,), {})()
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def main() -> int:
    fbc = load_first_block_cache()
    import comfy.ops

    # Every Linear comfy might hand a model, so the gate is tested against all of them.
    linears = {
        "disable_weight_init": comfy.ops.disable_weight_init.Linear(16, 16),
        "manual_cast": comfy.ops.manual_cast.Linear(16, 16),
    }
    print("comfy.ops Linear variants:")
    for name, linear in linears.items():
        print(f"  {name:<22} isinstance(nn.Linear)={isinstance(linear, torch.nn.Linear)!s:<5} "
              f"accepted={fbc._accepts_single_argument(linear)}")
    print(f"  {'TokenRefiner':<22} isinstance(nn.Linear)=False accepted="
          f"{fbc._accepts_single_argument(TokenRefiner())}\n")

    blocks = torch.nn.ModuleList
    expectations = []
    for linear_name, linear in linears.items():
        expectations += [
            (f"Flux-1 [{linear_name}]", "flux",
             make("Flux", double_blocks=blocks(), single_blocks=blocks(),
                  vector_in=None, txt_in=linear, params=Params())),
            # FLUX.2 rebuilds `vec` between the two loops, so it goes generic with the single
            # blocks left to the model. Asserted below that single_blocks_name is None.
            (f"Flux-2 global_mod [{linear_name}]", "generic",
             make("Flux", double_blocks=blocks(), single_blocks=blocks(),
                  vector_in=None, txt_in=linear, params=Params(global_modulation=True))),
            (f"Chroma no vector_in [{linear_name}]", "generic",
             make("Chroma", double_blocks=blocks(), single_blocks=blocks(),
                  txt_in=linear, params=Params())),
        ]
    expectations += [
        ("HunyuanVideo TokenRefiner", "generic",
         make("HunyuanVideo", double_blocks=blocks(), single_blocks=blocks(),
              vector_in=None, txt_in=TokenRefiner(), params=Params())),
        ("Hunyuan3D cond_in", "generic",
         make("Hunyuan3D", double_blocks=blocks(), single_blocks=blocks(),
              cond_in=torch.nn.Linear(16, 16), params=Params())),
        ("UNet", "unet",
         make("UNetModel", input_blocks=blocks(), middle_block=blocks(),
              output_blocks=blocks())),
        ("LTXVModel", "generic", make("LTXVModel", transformer_blocks=blocks())),
        ("NextDiT / Z-Image", "generic", make("NextDiT", layers=blocks())),
        ("MMDiT joint_blocks", "generic", make("OpenAISignatureMMDITWrapper",
                                               joint_blocks=blocks())),
        ("Wan (blocks)", "generic", make("WanModel", blocks=blocks())),
        # Its forward injects VACE blocks at indices of the main loop, which the wrapper's
        # one-iteration collapse would silently drop.
        ("VaceWan (index-interleaved)", "REFUSE",
         make("VaceWanModel", blocks=blocks(), vace_blocks=blocks())),
        # WanModel_S2V calls self.audio_injector(x, i, ...) after every block -- same hazard,
        # different attribute, and it would have slipped through a VACE-only guard.
        ("Wan S2V (audio injector)", "REFUSE",
         make("WanModel_S2V", blocks=blocks(), audio_injector=torch.nn.Identity())),
        # LTX-2.5: hidden state is (video, audio). Supported since the cache became part-wise;
        # tools/fbcache_tuple_test.py checks the arithmetic on a stand-in of the same shape.
        ("LTXAV (tuple hidden state)", "generic",
         make("LTXAVModel", transformer_blocks=torch.nn.ModuleList([AVBlock()]))),
        # LTX-2.3 and anything else without an annotation must NOT be refused on suspicion.
        ("LTXV 2.3 (unannotated)", "generic",
         make("LTXVModel", transformer_blocks=torch.nn.ModuleList([PlainBlock()]))),
        # Z-Image has no text stream at all and works today; a name-based guard would break it.
        ("Z-Image (no text stream)", "generic",
         make("NextDiT", layers=torch.nn.ModuleList([ZImageBlock()]))),
        ("nothing recognisable", "REFUSE", make("Mystery", widgets=blocks())),
    ]

    failures = 0
    print(f"{'model':<34} {'expected':<9} {'got':<9}")
    print("-" * 60)
    for name, expected, module in expectations:
        try:
            got = fbc.detect_model_route(module).kind
        except ValueError:
            got = "REFUSE"
        ok = got == expected
        failures += not ok
        print(f"{name:<34} {expected:<9} {got:<9} {'' if ok else '  <-- FAIL'}")

    # Two properties the `kind` alone does not capture.
    print("\nper-route details")
    print("-" * 60)
    checks = [
        ("FLUX.2 must leave single_blocks to the model",
         make("Flux", double_blocks=blocks(), single_blocks=blocks(), vector_in=None,
              txt_in=linears["manual_cast"], params=Params(global_modulation=True)),
         lambda r: r.single_blocks_name is None,
         lambda r: f"single_blocks_name={r.single_blocks_name!r}"),
        ("HunyuanVideo must still take single_blocks",
         make("HunyuanVideo", double_blocks=blocks(), single_blocks=blocks(), vector_in=None,
              txt_in=TokenRefiner(), params=Params()),
         lambda r: r.single_blocks_name == "single_blocks",
         lambda r: f"single_blocks_name={r.single_blocks_name!r}"),
        ("Wan blocks return hidden states only",
         make("WanModel", blocks=blocks()),
         lambda r: r.return_hidden_states_only,
         lambda r: f"return_hidden_states_only={r.return_hidden_states_only}"),
        ("a model with both keeps the specific name",
         make("Hybrid", transformer_blocks=blocks(), blocks=blocks()),
         lambda r: r.double_blocks_name == "transformer_blocks",
         lambda r: f"double_blocks_name={r.double_blocks_name!r}"),
        # `== "LTXVModel"` missed every subclass. Matching the MRO is what makes an LTXV
        # descendant inherit the single-stream setting instead of silently losing it.
        ("an LTXV subclass inherits return_only",
         make_subclass("LTXVModel", "SomeLTXVVariant",
                       transformer_blocks=torch.nn.ModuleList([PlainBlock()])),
         lambda r: r.return_hidden_states_only,
         lambda r: f"return_hidden_states_only={r.return_hidden_states_only}"),
    ]
    for label, module, predicate, describe in checks:
        route = fbc.detect_model_route(module)
        ok = predicate(route)
        failures += not ok
        print(f"{label:<45} {describe(route):<28} {'' if ok else '<-- FAIL'}")

    # detect_model_route raising is right for a pure function, but the node must not let that
    # reach the graph: an unsupported architecture should cost the acceleration, not the run.
    print("\nnode-level fallback")
    print("-" * 60)
    node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "Comfy-WaveSpeed-Fixed"
    sys.path.insert(0, str(node_dir.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "Comfy_WaveSpeed_Fixed", node_dir / "__init__.py",
        submodule_search_locations=[str(node_dir)])
    node_module = importlib.util.module_from_spec(spec)
    sys.modules["Comfy_WaveSpeed_Fixed"] = node_module
    spec.loader.exec_module(node_module)

    class FakePatcher:
        """The three ModelPatcher methods ApplyFBCacheOnModel touches."""

        def __init__(self, diffusion_model):
            self.diffusion_model = diffusion_model
            self.wrapper = None

        def clone(self):
            return self

        def get_model_object(self, name):
            return self.diffusion_model

        def set_model_unet_function_wrapper(self, wrapper):
            self.wrapper = wrapper

    node = node_module.NODE_CLASS_MAPPINGS["ApplyFBCacheOnModel"]()
    for label, module in (("unmapped architecture", make("Mystery", widgets=blocks())),
                          ("VaceWan", make("VaceWanModel", blocks=blocks(),
                                           vace_blocks=blocks()))):
        patcher = FakePatcher(module)
        try:
            returned = node.patch(patcher, "diffusion_model", 0.12)[0]
            applied = getattr(returned, "wrapper", None) is not None
            ok = returned is patcher and not applied
            print(f"{label:<28} returned model, wrapper attached={applied}   "
                  f"{'' if ok else '<-- FAIL'}")
            failures += not ok
        except Exception as error:
            print(f"{label:<28} RAISED {type(error).__name__} <-- FAIL "
                  f"(should degrade, not kill the graph)")
            failures += 1

    print(f"\n{'PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
