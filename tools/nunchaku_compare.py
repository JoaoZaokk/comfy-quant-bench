"""Measure one Z-Image checkpoint end to end, so two of them can be compared fairly.

The point of comparison here is SVDQuant W4A4 (Nunchaku) against the BF16 source of the same
model. That comparison is only meaningful if both runs see byte-identical inputs, so everything
that varies -- noise, conditioning, sampler, schedule, steps, seed -- is derived from the seed
and written into the result file alongside the timings. The caller compares the files; this
script never sees both models.

One model per process on purpose. The BF16 Z-Image is 12.3 GiB and the INT4 is 3.6 GiB; loading
both into one interpreter to "save a launch" is how a measurement turns into a measurement of
the allocator. Separate processes also mean a crash in one loader cannot corrupt the other's
numbers.

Peak VRAM is read from NVML, not from `torch.cuda.max_memory_allocated`. On the FLUX route the
whole transformer lives inside nunchaku's native engine and never becomes a torch tensor, so the
allocator-based figure read 0.21 GiB for a model whose real footprint is 6.71 GiB. Qwen and
Z-Image do not do this and the two figures agree there -- see `DevicePeak` for the measurements.
Both numbers are recorded and the source is written into the result file, so two runs measured
differently are refused rather than divided.

Conditioning is synthetic, at the width the model's own text projection expects. That makes the
quality columns meaningless -- a random context is not a prompt -- and the speed and memory
columns valid, which is what is being asked. Latents are saved so the two runs can be compared
numerically, not looked at.

**Exempt from `_timing.compare()`, and not from the lock.** `compare()` interleaves two paths in
one process, and the whole design here is one model per process -- there is no second path in this
interpreter to interleave against, and putting one there is the allocator measurement the first
paragraph refuses. What this file takes from `_timing` instead is `summarize()` (so the estimator
it uses is recorded in the file rather than described in a sentence) and `ratio_of()` (so every
ratio it prints carries its direction and its interval). It does take `BenchGuard`; see the bottom
of the file for why that was not optional.

    python tools/nunchaku_compare.py --model z_image_turbo_bf16.safetensors --loader comfy \
        --out bf16.pt
    python tools/nunchaku_compare.py --model svdq-int4_r32-z-image-turbo.safetensors \
        --loader nunchaku --out int4.pt
    python tools/nunchaku_compare.py --compare bf16.pt int4.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _timing import ratio_of, summarize  # noqa: E402


def _nvml_index_for_cuda(pynvml, ordinal: int) -> tuple[int, str]:
    """Map a CUDA ordinal to an NVML index by UUID. Returns (index, how it was resolved).

    Torch and NVML both report a `GPU-<uuid>` string for the same physical card, and that string
    is the only thing the two enumerations agree on -- the indices do not, whenever
    `CUDA_VISIBLE_DEVICES` is set or the driver orders devices differently from the runtime.

    Falls back to `ordinal` when torch cannot produce a UUID (older builds do not expose
    `properties.uuid`), and the fallback is **named in the return value** rather than silently
    assumed, so the result file records that this run's VRAM figure rests on an assumption. That
    is the round-1 lesson in this repo: a check that quietly degrades reads exactly like one that
    passed.
    """
    try:
        wanted = str(torch.cuda.get_device_properties(ordinal).uuid).lower().replace("gpu-", "")
    except Exception:
        return ordinal, "assumed ordinal == nvml index (torch reports no UUID)"
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        raw = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if raw.lower().replace("gpu-", "").strip() == wanted:
            return index, "matched by UUID"
    return ordinal, "assumed ordinal == nvml index (no UUID match found)"


class DevicePeak:
    """Peak GPU memory as the *driver* sees it, sampled from a background thread.

    `torch.cuda.max_memory_allocated` counts only what passed through PyTorch's caching
    allocator, and one nunchaku route keeps its weights somewhere else entirely. Measured on
    this host, all three routes, same tool, same day:

        route          quantised modules in graph    torch allocator    NVML device
        FLUX.1-dev                              2           0.21 GiB       6.71 GiB
        Qwen-Edit-2511                        480          12.35 GiB      12.60 GiB
        Z-Image                               204           3.91 GiB       4.55 GiB

    So it is not "the nunchaku path" -- it is FLUX specifically. That route loads the whole
    transformer into the native engine and hands back a wrapper, which is why only 2 quantised
    modules appear in the graph; the weights are never torch tensors and the allocator sees
    nothing but activations. Qwen and Z-Image build Python `SVDQW4A4Linear` modules whose
    buffers *are* torch tensors, and there the two figures agree to within a CUDA context.

    A 31x undercount is worse than a missing number: it printed next to a BF16 row that was
    correct, so the column read as a 31x memory win on the one model where the win is real but
    nothing like that large. The `quantised modules in graph` line this script already prints
    is the tell -- 2 means the footprint is invisible to PyTorch.

    NVML sees every byte on the card no matter who allocated it. That is the fix, and also
    the catch: on Windows the driver runs in WDDM mode and NVML will not break the total
    down per process (`nvmlDeviceGetComputeRunningProcesses` returns entries whose
    `usedGpuMemory` is unavailable), so device-wide usage is all there is. Subtracting a
    baseline taken immediately before the model loads makes it attributable again -- on the
    condition that nothing else on the card grows meanwhile. That condition is not assumed:
    the number of other compute processes is recorded and printed beside the figure, because
    on this host the GPU is regularly shared with unrelated LLM work.

    Polling, not a hook, because the allocation happens inside a CUDA extension there is no
    hook to place in. 50 ms is far below the seconds-scale phases being measured and costs a
    single NVML query per tick.
    """

    def __init__(self, index: int = 0, interval: float = 0.05):
        """`index` is a **CUDA ordinal**, and it is resolved to an NVML index by UUID.

        It used to be passed straight to `nvmlDeviceGetHandleByIndex`, which is only correct when
        the two numbering schemes happen to agree. Under `CUDA_VISIBLE_DEVICES=1` the CUDA ordinal
        is 0 and the NVML index is 1, so every VRAM figure in the result file would have described
        the *other* card while naming this one -- a silent, plausible number, which is the worst
        kind. There are two cards in this box (3090 on nvml0, 3080 Ti on nvml1) and CLAUDE.md
        records a whole measurement session that ran on the 3080 Ti while the lock sat on an idle
        3090, found only because the probe cache stamped the adapter name into every line.

        So: match on UUID, and record `device_name` and `device_match` in every record this file
        writes. Stamping the name is the cheapest insurance there is, and it is what exposed the
        wrong-card session in the sibling repo.
        """
        self.interval = interval
        self.baseline = 0
        self.peak = 0
        self.min_after_peak = 0
        self.other_processes = -1
        self.name = None
        self.nvml_index = None
        self.match = "not resolved"
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self.nvml_index, self.match = _nvml_index_for_cuda(pynvml, index)
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.nvml_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
            print(f"NVML: cuda:{index} -> nvml{self.nvml_index} {self.name} ({self.match})")
        except Exception as exc:
            print(f"NVML unavailable ({type(exc).__name__}: {exc}). Peak VRAM will come from "
                  f"the torch allocator alone, which undercounts any extension that calls "
                  f"cudaMalloc itself -- on the nunchaku path that is most of the model.")
            self._nvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def _used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def start(self) -> None:
        if not self.available:
            return
        self.baseline = self.peak = self.min_after_peak = self._used()
        try:
            procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            # Count only processes NVML can actually attribute memory to. On WDDM it attributes
            # none of them, and the raw count is worthless as a "who else is computing" signal:
            # measured on this host with an idle 3090 at 36 MiB, NVML still listed four compute
            # processes -- the System process, a vendor service and a tray app -- every one with
            # usedGpuMemory None. Warning on that count would fire on every single run.
            attributed = [p for p in procs if getattr(p, "usedGpuMemory", None)]
            self.per_process_available = bool(attributed) or not procs
            self.other_processes = max(0, len(attributed) - 1) if attributed else 0
            self.listed_processes = len(procs)
        except Exception:
            self.other_processes = -1
            self.listed_processes = -1
            self.per_process_available = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                used = self._used()
                # The running minimum seen *since the current peak was set*, not since start:
                # it resets to the new peak whenever one is hit, so it falls when memory is freed
                # after that peak and tracks a fresh peak again once one occurs. A max-only figure
                # can never report that anything was freed at all; a caller reading only `peak`
                # has no way to tell "nothing was released" from "release was never measured".
                if used >= self.peak:
                    self.peak = used
                    self.min_after_peak = used
                else:
                    self.min_after_peak = min(self.min_after_peak, used)
            except Exception:
                return  # a dead sampler is reported as a missing number, not a wrong one

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self.available:
            # `device_name` is still emitted, from torch, even with no NVML: which card a run used
            # is the fact that has to survive into the record file no matter what else failed.
            try:
                fallback = torch.cuda.get_device_name(torch.cuda.current_device())
            except Exception:
                fallback = None
            return {"device_peak_gib": None, "device_baseline_gib": None,
                    "device_released_gib": None, "other_gpu_processes": None,
                    "device_name": fallback, "nvml_index": None,
                    "device_match": "NVML unavailable; name read from torch"}
        # Explicitly a number, never omitted: `peak - min_after_peak` is 0.0 when nothing was
        # freed after the peak sample, and 0.0 is a real, measured result -- not the same thing
        # as "release was not measured" (which is `None`, the NVML-unavailable case above). A
        # caller that tests this value with a bare `if released:` will silently drop the
        # legitimate zero; test `is not None` instead.
        return {
            "device_peak_gib": (self.peak - self.baseline) / 2**30,
            "device_baseline_gib": self.baseline / 2**30,
            "device_released_gib": (self.peak - self.min_after_peak) / 2**30,
            "other_gpu_processes": self.other_processes,
            "per_process_attribution": self.per_process_available,
            "device_name": self.name,
            "nvml_index": self.nvml_index,
            "device_match": self.match,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="file name inside models/diffusion_models")
    parser.add_argument("--loader", choices=["comfy", "nunchaku"], default="comfy",
                        help="`comfy` is comfy.sd.load_diffusion_model, the normal path. "
                             "`nunchaku` goes through NunchakuZImageDiTLoader, which is the only "
                             "way to read an SVDQuant checkpoint -- the tensor layout is not "
                             "something the stock loader can interpret.")
    parser.add_argument("--out", help="write the latent and the measurements here")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="report on two files written by earlier runs and exit")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=0,
                        help="override the text-stream width when the model will not report it. "
                             "Guessing this wrong fails inside the text projection, not in the "
                             "block stack, so the script asks rather than assumes.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--clip-type", default="stable_diffusion",
                        help="which CLIPType to build. FLUX needs `flux`, and passing the wrong "
                             "one produces an encoder that loads and then feeds the model a "
                             "context of the wrong width.")
    parser.add_argument("--clip", nargs="+", help="text encoder(s) in models/text_encoders. FLUX "
                                       "takes two -- clip_l and t5xxl -- so this accepts a list. "
                                       "Giving this "
                                       "switches off synthetic conditioning and runs the real "
                                       "pipeline, which is the only version whose output can be "
                                       "judged on quality rather than only on speed.")
    parser.add_argument("--vae", help="VAE in models/vae. Required with --clip: without a decode "
                                      "there is no image to look at, and latents are not a "
                                      "quality judgement.")
    parser.add_argument("--prompt", default="a red apple on a weathered wooden table, "
                                            "afternoon light, sharp detail")
    parser.add_argument("--negative", default="")
    parser.add_argument("--png", help="write the decoded image here (implies --clip and --vae)")
    parser.add_argument("--attention", choices=["default", "sage", "sparge", "flash"],
                        default="default",
                        help="which attention backend to force. `sage` is SageAttention 2 -- on "
                             "sm86 that is the INT8-QK/FP16-PV path; 2++ is an sm89+ branch and "
                             "is not reachable on Ampere. `sparge` is SpargeAttn, which is not a "
                             "layer on top of Sage but a replacement containing it: the entry "
                             "points are named spas_sage2_attn_*, Sage2 quantisation plus block "
                             "sparsity. Triton is not a third option -- it runs inside both, "
                             "doing the INT8 quantisation via per_thread_int8_triton.")
    parser.add_argument("--sparge-topk", type=float, default=0.5,
                        help="fraction of blocks SpargeAttn keeps. Lower skips more. Like "
                             "FBCache's threshold this is not portable between models.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed passes after the discarded warm-up. The reported figure is "
                             "the fastest, because the slow ones are slow for reasons that have "
                             "nothing to do with the model.")
    return parser.parse_args()


def report(a_path: str, b_path: str) -> int:
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)

    # Same prompt string is not enough: two runs can tokenize identically and still differ if
    # one of them fell back to synthetic conditioning. Compare the norm of the encoded tensor.
    #
    # `device_name` joins that list because two runs on two different cards are not a comparison
    # of two checkpoints, and this box has a 3090 and a 3080 Ti in it. CLAUDE.md records a whole
    # sibling session whose work landed on the 3080 Ti while the lock sat on an idle 3090; a
    # result file that does not say which card it used cannot rule that out afterwards. Files
    # written before this key existed simply skip the check (the `not in` guard below), so old
    # comparisons still work -- they just cannot prove they were on one card.
    for key in ("shape", "steps", "seed", "cfg", "context", "prompt", "cond_norm", "device_name"):
        if key not in a["meta"] or key not in b["meta"]:
            continue
        if a["meta"][key] != b["meta"][key]:
            print(f"!! the two runs disagree on {key}: {a['meta'][key]} vs {b['meta'][key]}. "
                  f"Nothing below is a comparison.")
            return 1

    print(f"\n{'=' * 78}")
    print(f"shape {a['meta']['shape']}, {a['meta']['steps']} steps, seed {a['meta']['seed']}, "
          f"cfg {a['meta']['cfg']}\n")
    head = f"{'':<26}{'A':>14}{'B':>14}{'B vs A':>14}"
    print(head)
    print(f"{'model':<26}{Path(a['meta']['model']).stem[:13]:>14}"
          f"{Path(b['meta']['model']).stem[:13]:>14}")
    print(f"{'attention':<26}{a['meta'].get('attention', '?'):>14}"
          f"{b['meta'].get('attention', '?'):>14}")
    if a["meta"].get("mean_sparsity") or b["meta"].get("mean_sparsity"):
        print(f"{'mean blocks skipped':<26}{str(a['meta'].get('mean_sparsity')):>14}"
              f"{str(b['meta'].get('mean_sparsity')):>14}")

    def row(label, ka, kb, fmt="{:.2f}", ratio=True, lower_is_better=False, spread=None,
            exact=False):
        # Disk and VRAM are better when smaller, so B/A reads backwards: a genuine 3.4x saving
        # printed as "0.29x" looks like a regression at a glance. Those rows report how many
        # times LIGHTER B is, and say so, rather than leaving the reader to invert it.
        #
        # The direction logic itself now lives in `_timing.Ratio`, which is the only place in the
        # tree that formats a ratio: `exact=True` marks a quantity with no burst-to-burst spread
        # (bytes on disk do not vary between passes) so that it is not given a fictitious
        # interval, and `spread=` supplies a real one where there is one.
        va, vb = ka, kb
        cell = f"{fmt.format(va):>14}{fmt.format(vb):>14}"
        if ratio and va and vb:
            # Both directions, spelled out. "0.81x less" for a row where B got WORSE would be a
            # lie dressed as a saving, so a regression is reported as "1.24x more".
            if lower_is_better:
                cell += f"{ratio_of(va, vb, better='less', worse='more', series=spread, exact=exact):>26}"
            else:
                cell += f"{ratio_of(vb, va, better='higher', worse='lower', series=spread, exact=exact):>26}"
        print(f"{label:<26}{cell}")

    # `exact=True` on disk alone: a file size is the same on every pass, so a bracket on it would
    # be invented. Every other row below is a measurement taken once and gets "(1 burst, no
    # interval)" instead -- which is the honest thing to print about load time, a quantity this
    # tool measures exactly once per process and never repeats.
    row("disk GiB", a["meta"]["disk"] / 2**30, b["meta"]["disk"] / 2**30,
        lower_is_better=True, exact=True)
    row("load seconds", a["meta"]["load_s"], b["meta"]["load_s"], lower_is_better=True)
    # A run measured with NVML and a run measured with the torch allocator are not the same
    # quantity, and putting them in one ratio is how the original 0.21 GiB got believed. Files
    # written before this column had a source recorded are torch-allocator files by definition.
    src_a = a["meta"].get("peak_source", "torch caching allocator (undercounts CUDA extensions)")
    src_b = b["meta"].get("peak_source", "torch caching allocator (undercounts CUDA extensions)")
    if src_a != src_b:
        print(f"{'peak VRAM GiB':<26}{a['meta']['peak_gib']:>14.2f}"
              f"{b['meta']['peak_gib']:>14.2f}{'--':>12}")
        print(f"  !! not comparable: A measured by {src_a}, B by {src_b}. Re-run both.")
    else:
        row("peak VRAM GiB", a["meta"]["peak_gib"], b["meta"]["peak_gib"],
            lower_is_better=True)
        if "torch" in src_a and "undercounts" in src_a:
            print("  (torch allocator only -- a nunchaku run's real footprint is larger than "
                  "this)")
    if a["meta"].get("torch_peak_gib") is not None and b["meta"].get("torch_peak_gib") is not None:
        row("  of which torch alloc", a["meta"]["torch_peak_gib"], b["meta"]["torch_peak_gib"],
            ratio=False)
    # `is not None`, always -- never `if rel_a:` -- because 0.00 GiB released is a real,
    # measured result (peak was never given back) and must print as "0.00", not be dropped the
    # way a bare truthiness check would drop it. `None` (NVML was unavailable on that run) is the
    # only case that prints "not measured", and each side is judged independently: one run can
    # have NVML and the other not.
    rel_a = a["meta"].get("device_released_gib")
    rel_b = b["meta"].get("device_released_gib")
    if rel_a is not None and rel_b is not None:
        row("  VRAM released after peak", rel_a, rel_b, ratio=False)
    elif rel_a is not None or rel_b is not None:
        print(f"{'  VRAM released after peak':<26}"
              f"{(f'{rel_a:.2f}' if rel_a is not None else 'not measured'):>14}"
              f"{(f'{rel_b:.2f}' if rel_b is not None else 'not measured'):>14}")
    others = [m["meta"].get("other_gpu_processes") for m in (a, b)]
    if any(o for o in others if o and o > 0):
        print(f"  !! other GPU processes during measurement: A={others[0]}, B={others[1]}. "
              f"The VRAM row includes them.")
    # A **bound**, not a paired spread. `_timing.compare()` can pair a ratio burst by burst
    # because it interleaves both paths in one process; A and B here are two separate processes
    # measured minutes apart -- one model per process, on purpose, because loading a 12.3 GiB BF16
    # and a 3.6 GiB INT4 into one interpreter measures the allocator instead of the models. So the
    # widest and narrowest ratios consistent with the two pass sets are what can honestly be
    # shown, and they are wider than a paired interval would be.
    passes_a, passes_b = a["meta"].get("passes") or [], b["meta"].get("passes") or []
    speed_spread = None
    if len(passes_a) > 1 and len(passes_b) > 1:
        speed_spread = [min(passes_a) / max(passes_b), max(passes_a) / min(passes_b)]
    row("seconds / sampling pass", a["meta"]["best_s"], b["meta"]["best_s"],
        lower_is_better=True, spread=speed_spread)
    row("seconds / step", a["meta"]["best_s"] / a["meta"]["steps"],
        b["meta"]["best_s"] / b["meta"]["steps"], "{:.4f}", lower_is_better=True,
        spread=speed_spread)
    # Same fact as the row above, and it used to be printed as a raw a/b ratio: a B that is 24%
    # slower came out as "speedup 0.81x", which reads as a saving. The direction rule lives in
    # `_timing.Ratio` now, so this line and the rows above cannot drift apart again.
    verdict = ratio_of(a["meta"]["best_s"], b["meta"]["best_s"], series=speed_spread,
                       better="faster", worse="slower")
    print(f"{'B vs A overall':<26}{'':>14}{'':>14}{verdict:>20}")
    if speed_spread:
        print("  (that interval is a BOUND from the two pass sets, not a paired per-burst spread: "
              "A and B\n   ran in separate processes and cannot be interleaved)")

    la, lb = a["latent"].float(), b["latent"].float()
    if la.shape == lb.shape:
        rel = ((la - lb).norm() / lb.norm().clamp(min=1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            la.flatten().unsqueeze(0), lb.flatten().unsqueeze(0)).item()
        print(f"\nlatent relL2 A vs B : {rel:.4f}")
        print(f"latent cosine       : {cos:.4f}")
        if a["meta"].get("prompt"):
            print(f"\nprompt: {a['meta']['prompt']!r}")
            print("Both runs used the same encoded conditioning -- the |cond| check above would "
                  "have refused\notherwise -- so the divergence is the quantisation, not the "
                  "input. It is still not a quality\nverdict: two different trajectories can "
                  "both be good. Look at the decoded images.")
        else:
            print("\nConditioning is random, so these two numbers say how far the quantised "
                  "model moved the\ntrajectory -- not whether the image is worse. Judging "
                  "quality needs a real prompt through a\nreal text encoder and a look at the "
                  "pixels.")
    else:
        print(f"\nlatent shapes differ: {list(la.shape)} vs {list(lb.shape)} -- not comparable")

    # The estimator, named. Three other timing tools in `tools/` reduce with a median and this one
    # reduces with `min`, which is a defensible choice for a whole-model sampling run -- the floor
    # is "nothing else on the card" and there is no ceiling -- but a table that mixes the two
    # without saying so cannot be read as one measurement. `_timing.ESTIMATORS` carries both and
    # this line says which was used, on every run.
    est_a = a["meta"].get("estimator", "min")
    est_b = b["meta"].get("estimator", "min")
    print(f"\nESTIMATOR: {est_a} of {a['meta']['repeats']} passes after a discarded warm-up "
          f"(B: {est_b}). Not a median -- the other timing tools in tools/ use one, and these "
          f"columns are not\ninterchangeable with theirs.")
    if est_a != est_b:
        print(f"!! A and B were reduced differently ({est_a} vs {est_b}). The ratio above divides "
              f"two different estimators.")
    per = a["meta"]["passes"], b["meta"]["passes"]
    for tag, side in (("A", a), ("B", b)):
        print(f"{tag} device: {side['meta'].get('device_name', 'not recorded')} "
              f"({side['meta'].get('device_match', 'no mapping recorded')})")
    print(f"A passes: {', '.join(f'{x:.2f}' for x in per[0])}")
    print(f"B passes: {', '.join(f'{x:.2f}' for x in per[1])}")
    spread = [max(p) / min(p) for p in per if p and min(p)]
    if spread and max(spread) > 1.15:
        # A dispersion factor, not a comparison: it is max/min of one run's own passes, always
        # >= 1, and there is no "faster"/"slower" to name. It still goes through `Ratio` --
        # `exact=True`, both direction words blank -- so that grepping `tools/` for a bare
        # `:.2f}x` finds every hit inside `_timing.py` and nowhere else. That grep is the
        # closing check for this rule and a mechanical check is the only kind that survives the
        # next session.
        worst = ratio_of(max(spread), 1.0, better="", worse="", exact=True)
        print(f"!! the passes within one run spread by {worst}, which is large enough "
              f"to compete\n   with the difference being measured. Treat the comparison as "
              f"indicative only.")
    return 0


def load_model(args, folder_paths, comfy_sd):
    """Return (model, seconds). Two loaders, because SVDQuant is not a format comfy can read."""
    candidate = Path(args.model)
    path = str(candidate) if candidate.is_file() else \
        folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    print(f"loading [{args.loader}] {path}", flush=True)
    started = time.perf_counter()
    if args.loader == "comfy":
        model = comfy_sd.load_diffusion_model(path)
    else:
        # The node package directory has a dash, so it cannot be imported by name. ComfyUI loads
        # it by path at boot; do the same rather than reimplementing the SVDQuant reader here --
        # the point is to measure the path a user actually gets, not a private copy of it.
        import importlib.util

        node_dir = PORTABLE_ROOT / "ComfyUI" / "custom_nodes" / "ComfyUI-nunchaku"
        sys.path.insert(0, str(node_dir.parent))
        spec = importlib.util.spec_from_file_location(
            "ComfyUI_nunchaku", node_dir / "__init__.py",
            submodule_search_locations=[str(node_dir)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["ComfyUI_nunchaku"] = module
        spec.loader.exec_module(module)
        # The loaders do not share a signature. ZImage and QwenImage take a bare model name;
        # FLUX takes six arguments, one of which is `cache_threshold` -- Nunchaku's own
        # first-block cache. It is pinned to 0 here: leaving it on would fold a caching speedup
        # into a number that is supposed to isolate quantisation.
        name = Path(path).name
        if "flux" in name.lower():
            loader = module.NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"]()
            model = loader.load_model(
                model_path=name, attention="nunchaku-fp16", cache_threshold=0,
                cpu_offload="disable", device_id=0, data_type="bfloat16")[0]
        elif "qwen" in name.lower():
            # Qwen's loader has its own signature again: cpu_offload is positional and required,
            # and num_blocks_on_gpu only matters when offloading is on. "disable" keeps the whole
            # model resident, which is the only setting under which a timing means anything.
            loader = module.NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"]()
            model = loader.load_model(name, cpu_offload="disable",
                                      num_blocks_on_gpu=1, use_pin_memory="disable")[0]
        else:
            loader = module.NODE_CLASS_MAPPINGS["NunchakuZImageDiTLoader"]()
            model = loader.load_model(name)[0]
    return model, time.perf_counter() - started, Path(path)


def main() -> int:
    args = parse_args()
    if args.compare:
        return report(*args.compare)
    if not args.model or not args.out:
        print("--model and --out are required unless --compare is given")
        return 2

    # The backend has to be chosen before anything imports comfy.ldm.modules.attention, because
    # that module picks `optimized_attention` once at import time and every block module then
    # binds the chosen function into its own namespace. Flipping a flag afterwards patches a
    # name nobody reads any more.
    import comfy.cli_args
    if args.attention in ("sage", "sparge"):
        comfy.cli_args.args.use_sage_attention = True
    elif args.attention == "flash":
        # This routes to the standalone `flash_attn` package via
        # comfy/ldm/modules/attention.py:795, NOT to torch's SDPA flash backend. The Windows
        # torch build here reports "Torch was not compiled with flash attention", which is true
        # and irrelevant: ComfyUI never calls that path.
        comfy.cli_args.args.use_flash_attention = True

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import comfy.utils
    import folder_paths

    # ComfyUI reads extra_model_paths.yaml from main.py, not from folder_paths import time. A
    # bare script therefore cannot see the second drive at all, and every model mounted from
    # D:/ComfyUI-Models raises "not found" -- including, silently, inside the Nunchaku loader
    # nodes, which resolve names the same way.
    extra = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        import utils.extra_config
        utils.extra_config.load_extra_path_config(str(extra))

    sparsity_seen = []
    if args.attention == "sparge":
        import comfy.ldm.modules.attention as comfy_attention
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

        if not comfy_attention.SAGE_ATTENTION_IS_AVAILABLE:
            print("sage attention did not load, so the sparge shim has nothing to sit in")
            return 1

        # `attention_sage` calls the module global `sageattn`, so replacing that name reaches
        # every caller regardless of who imported `optimized_attention` and when. The kwargs do
        # not line up: ComfyUI passes `sm_scale` and `smooth_k`, SpargeAttn wants `scale`.
        # Two things SpargeAttn cannot take, both of which ComfyUI hands it as a matter of course:
        #
        #   assert q.size(-2)>=128, "seq_len should be not less than 128."   (core.py:154)
        #
        # every cross-attention call here carries 23 text tokens, and there is nothing to
        # sparsify in 23 tokens anyway; and `smooth_k=False`, which ComfyUI passes, hits an
        # upstream bug -- core.py:163 assigns `km` only inside `if smooth_k:` and core.py:169
        # uses it unconditionally, so the call dies with UnboundLocalError. Both cases go to
        # Sage rather than to PyTorch: falling back to the slowest path would make this
        # measurement look like a SpargeAttn result when it is not one.
        def sparge_shim(q, k, v, sm_scale=None, tensor_layout="HND", is_causal=False,
                        smooth_k=False, attn_mask=None, **_ignored):
            seq = q.size(-2) if tensor_layout == "HND" else q.size(-3)
            if attn_mask is not None or seq < 128:
                return comfy_attention_sageattn(
                    q, k, v, sm_scale=sm_scale, tensor_layout=tensor_layout,
                    is_causal=is_causal, smooth_k=smooth_k, attn_mask=attn_mask)
            out, sparsity = spas_sage2_attn_meansim_topk_cuda(
                q, k, v, scale=sm_scale, tensor_layout=tensor_layout, is_causal=is_causal,
                smooth_k=True, topk=args.sparge_topk, output_dtype=q.dtype,
                return_sparsity=True)
            sparsity_seen.append(float(sparsity))
            return out

        comfy_attention_sageattn = comfy_attention.sageattn
        comfy_attention.sageattn = sparge_shim
        print(f"attention: SpargeAttn (topk {args.sparge_topk}) replacing sageattn", flush=True)
    elif args.attention == "sage":
        # Same idiom as the sparge and flash branches just above/below: check the module's own
        # availability flag before claiming the backend is in use. Without this, an environment
        # where SageAttention failed to import silently falls back to `attention_pytorch` (see
        # attention.py:646,676-680) and the run gets labelled "sage" anyway -- a mislabelled
        # measurement, not a missing one.
        import comfy.ldm.modules.attention as comfy_attention
        if not comfy_attention.SAGE_ATTENTION_IS_AVAILABLE:
            print("sage attention did not load, so this would silently measure the default")
            return 1
        print("attention: SageAttention", flush=True)
    elif args.attention == "flash":
        import comfy.ldm.modules.attention as comfy_attention
        if not comfy_attention.FLASH_ATTENTION_IS_AVAILABLE:
            print("flash_attn is not importable, so this would silently measure the default")
            return 1
        print("attention: FlashAttention (standalone flash_attn package)", flush=True)
    else:
        print("attention: ComfyUI default", flush=True)

    # Without this the weights stream back per step and the transfer, not the arithmetic, is what
    # gets timed -- and the quantised model would look faster purely because it is smaller to
    # move. That would be a real user-visible effect, but it is not the effect under test.
    comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
    comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM
    device = comfy.model_management.get_torch_device()
    print(f"device {torch.cuda.get_device_name(0)} cc {torch.cuda.get_device_capability(0)}, "
          f"{comfy.model_management.get_free_memory(device) / 2**30:.1f} GiB free", flush=True)

    # O encoder tem de sair da VRAM antes do modelo de difusao entrar. Qwen-2.5-VL-7B
    # ocupa ~15 GiB e o Qwen-Image INT4 outros 12: juntos estouram os 24 GiB da placa,
    # e o OOM aparece no meio da amostragem, longe da causa.
    positive = negative = None
    cond_shape = None
    cond_fingerprint = 0.0
    if args.clip:
        # The real pipeline. Encode first and drop the encoder before the diffusion model is
        # touched: Qwen3-4B is ~8 GiB and holding it alongside a 12 GiB BF16 transformer is how
        # a "peak VRAM" column stops describing the model under test. The encoded conditioning
        # is saved into the result file so the other run can be checked for having used exactly
        # the same tensors rather than merely the same prompt string.
        clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
        print(f"loading text encoder(s) {[Path(c).name for c in clip_paths]} "
              f"as {args.clip_type}", flush=True)
        clip = comfy.sd.load_clip(
            ckpt_paths=clip_paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=getattr(comfy.sd.CLIPType, args.clip_type.upper()))
        positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))[0])]
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
        positive[0][0] = positive[0][0].clone().cpu()
        negative[0][0] = negative[0][0].clone().cpu()
        cond_shape = list(positive[0][0].shape)
        cond_fingerprint = float(positive[0][0].float().norm())
        print(f"encoded prompt: {cond_shape}, |cond| {cond_fingerprint:.4f}", flush=True)
        del clip
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # Baseline goes here, after the encoder is gone and before a single weight of the model
    # under test is read, so that what the sampler reports afterwards is this model's cost and
    # not the encoder's leftovers.
    peak = DevicePeak(index=torch.cuda.current_device())
    peak.start()
    if peak.available:
        # Warn on resident memory, not on the process count: what threatens the subtraction is
        # something big already on the card, and the count is unattributable here anyway. A
        # baseline this size is roughly a bare CUDA context plus a desktop compositor.
        baseline_gib = peak.baseline / 2**30
        if baseline_gib > 0.5:
            print(f"warning: {baseline_gib:.2f} GiB already resident on this GPU before the "
                  f"model loads. It is subtracted from the figure below, which is only correct "
                  f"if it does not grow during the run.", flush=True)
        if peak.other_processes > 0:
            print(f"warning: {peak.other_processes} other process(es) with attributable GPU "
                  f"memory. The device-wide VRAM figure cannot separate them from this run.",
                  flush=True)

    try:
        model, load_s, path = load_model(args, folder_paths, comfy.sd)
    except Exception:
        print("LOAD FAILED")
        traceback.print_exc()
        return 1
    print(f"loaded in {load_s:.1f}s", flush=True)

    diffusion_model = model.get_model_object("diffusion_model")
    print(f"diffusion model class: {type(diffusion_model).__name__}")
    # Whether the SVDQuant kernels are actually in the graph is not something to take on trust:
    # a loader that silently dequantised to BF16 would produce a working model with none of the
    # speed, and the timings would then be comparing BF16 to BF16.
    quant_layers = [type(m).__name__ for m in diffusion_model.modules()
                    if "Nunchaku" in type(m).__name__ or "SVDQ" in type(m).__name__]
    if quant_layers:
        seen = {}
        for name in quant_layers:
            seen[name] = seen.get(name, 0) + 1
        print(f"quantised modules in graph: "
              f"{', '.join(f'{k}×{v}' for k, v in sorted(seen.items()))}")
    else:
        print("quantised modules in graph: none found "
              "(for the nunchaku loader this would mean the checkpoint was not read as SVDQuant)")

    latent_format = model.model.latent_format
    channels = latent_format.latent_channels
    side = max(args.size // 8, 8)
    shape = [1, channels, side, side]
    # Qwen-Image carries a temporal axis even for stills: process_img unpacks
    # `bs, c, t, h, w = x.shape` and a 4-D latent dies with "not enough values to unpack".
    if "qwen" in str(args.model).lower():
        shape = [1, channels, 1, side, side]

    # Ask the model how wide its text stream is instead of assuming. Z-Image is a Lumina/NextDiT
    # derivative whose cap_embedder is an RMSNorm of a fixed width, and feeding it anything else
    # fails inside the embedder before the transformer is ever reached -- which is exactly how an
    # earlier version of this measurement died at 4096 against a model that wanted 2560.
    context_dim = None
    for attr in ("cap_feat_dim", "cap_dim", "caption_channels", "context_in_dim"):
        context_dim = context_dim or getattr(diffusion_model, attr, None)
    if context_dim is None:
        embedder = getattr(diffusion_model, "cap_embedder", None)
        if embedder is not None:
            for sub in embedder.modules():
                shape_attr = getattr(sub, "normalized_shape", None)
                if shape_attr:
                    context_dim = int(shape_attr[0])
                    break
    if context_dim is None:
        # FLUX hides it: the Nunchaku loader returns a ComfyFluxWrapper, which exposes none of
        # the attributes above. The width lives on the params object the real model was built
        # from, so look there before giving up.
        for owner in (diffusion_model, getattr(diffusion_model, "model", None), model.model):
            params = getattr(owner, "params", None)
            if params is not None and getattr(params, "context_in_dim", None):
                context_dim = int(params.context_in_dim)
                break
    if context_dim is None and args.context_dim:
        context_dim = args.context_dim
    # Only the synthetic path needs this. With a real encoder the width comes from the encoder,
    # and refusing to run because the model would not self-report is a check firing on a
    # question nobody asked.
    if context_dim is None and not args.clip:
        print("could not determine the model's context width; refusing to guess. "
              "Pass --context-dim if you know it (FLUX.1 is 4096, Z-Image 2560).")
        return 1
    context_dim = int(context_dim) if context_dim else 0

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.zeros(shape, device="cpu")

    if args.clip:
        pass  # ja codificado antes de carregar o modelo de difusao
    else:
        context = torch.randn([1, args.context_tokens, context_dim],
                              generator=generator, dtype=torch.float32) * 0.02
        pooled = torch.zeros([1, 768], dtype=torch.float32)
        positive = [[context, {"pooled_output": pooled}]]
        negative = [[torch.zeros_like(context), {"pooled_output": pooled.clone()}]]
        cond_shape = [1, args.context_tokens, context_dim]
        cond_fingerprint = float(context.norm())
    print(f"latent {shape}, context {cond_shape}\n", flush=True)

    def one_pass(steps):
        noise = comfy.sample.prepare_noise(latent, args.seed, None)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = comfy.sample.sample(
            model, noise, steps, args.cfg, "euler", "simple",
            positive, negative, latent,
            denoise=1.0, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None, callback=None,
            disable_pbar=True, seed=args.seed)
        torch.cuda.synchronize()
        return out, time.perf_counter() - started

    try:
        one_pass(2)  # discarded: absorbs kernel autotuning and allocator growth
    except Exception:
        print("SAMPLING FAILED (warm-up)")
        traceback.print_exc()
        return 1

    torch.cuda.reset_peak_memory_stats()
    passes, out = [], None
    for i in range(args.repeats):
        out, elapsed = one_pass(args.steps)
        passes.append(elapsed)
        print(f"pass {i + 1}: {elapsed:.2f}s for {args.steps} steps "
              f"({elapsed / args.steps:.4f} s/step)", flush=True)
    peak.stop()
    torch_peak_gib = torch.cuda.max_memory_allocated() / 2**30
    device = peak.summary()
    # Prefer the driver's number. It is the only one that sees the nunchaku extension's own
    # allocations, and where both are trustworthy it is the larger of the two anyway.
    peak_gib = device["device_peak_gib"]
    if peak_gib is None:
        peak_gib = torch_peak_gib

    # `best_s` and the recorded estimator come from the same call, so a future change to one
    # cannot leave the other describing a reduction that no longer happens. `min` is kept
    # deliberately -- see the ESTIMATOR line in `report()` for why -- but it is now named in the
    # file rather than only in a sentence at the bottom of the output.
    best_s, _passes_lo, _passes_hi = summarize(passes, estimator="min")
    meta = {
        "model": args.model, "loader": args.loader, "disk": path.stat().st_size,
        "load_s": load_s, "peak_gib": peak_gib, "passes": passes, "best_s": best_s,
        "estimator": "min",
        "peak_source": "nvml device-wide minus baseline" if device["device_peak_gib"] is not None
                       else "torch caching allocator (undercounts CUDA extensions)",
        "torch_peak_gib": torch_peak_gib, **device,
        "repeats": args.repeats, "shape": shape, "steps": args.steps, "seed": args.seed,
        "cfg": args.cfg, "context": cond_shape, "prompt": args.prompt if args.clip else None,
        "cond_norm": round(cond_fingerprint, 4),
        "quant_modules": len(quant_layers), "attention": args.attention,
        "sparge_topk": args.sparge_topk if args.attention == "sparge" else None,
        "mean_sparsity": (round(sum(sparsity_seen) / len(sparsity_seen), 4)
                          if sparsity_seen else None),
    }
    if sparsity_seen:
        # A sparse kernel that reports zero skipped blocks is a slow dense kernel wearing a hat.
        print(f"SpargeAttn skipped a mean of {meta['mean_sparsity']:.4f} of blocks "
              f"across {len(sparsity_seen)} attention calls")

    if args.png:
        if not args.vae:
            print("--png needs --vae")
            return 2
        # Free the transformer before the VAE arrives. Peak VRAM was already recorded above, so
        # nothing is lost by this, and on the BF16 run the decode would otherwise have to fit
        # beside 12 GiB of weights.
        del model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        vae_path = folder_paths.get_full_path_or_raise("vae", args.vae)
        print(f"loading VAE {vae_path}", flush=True)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        image = vae.decode(out.to(vae.device if hasattr(vae, "device") else "cpu").float())
        # Qwen-Image decodes to (B, T, H, W, C) because its latent carries a temporal axis even
        # for a still, so a fixed 4-D assumption ends in
        # "TypeError: Cannot handle this data type: (1, 1, 3, 1)". Squeeze the leading axes until
        # what is left is a single (H, W, C) frame, whatever the model's convention.
        array = image
        while array.ndim > 3:
            array = array[0]
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = array.movedim(0, -1)
        from PIL import Image
        Image.fromarray(
            (array.detach().clamp(0, 1).float().cpu().numpy() * 255).round().astype("uint8")
        ).save(args.png)
        print(f"decoded image -> {args.png}")
        meta["png"] = args.png

    torch.save({"latent": out.float().cpu(), "meta": meta}, args.out)
    print(f"\nbest {best_s:.2f}s (estimator: min of {len(passes)} passes), "
          f"peak VRAM {peak_gib:.2f} GiB "
          f"({meta['peak_source']}) -> {args.out}")
    if device["device_peak_gib"] is not None:
        # Printing both is the point. On the comfy path they should be within a few hundred MiB
        # of each other; a large gap on the nunchaku path is the extension's allocations, and
        # a large gap on the comfy path would mean something else grew on the card mid-run.
        print(f"  torch allocator saw {torch_peak_gib:.2f} GiB; driver saw "
              f"{peak_gib:.2f} GiB above a {device['device_baseline_gib']:.2f} GiB baseline "
              f"with {device['other_gpu_processes']} other process(es) present")
    # `is not None`, not truthy: 0.00 GiB released is a real, measured result (nothing was freed
    # after the peak) and must print as that, not vanish the way a bare `if device[...]:` would
    # make it vanish. `None` -- NVML unavailable -- is the only case that prints "not measured".
    released = device.get("device_released_gib")
    if released is not None:
        print(f"  {released:.2f} GiB released after peak (measured; a max-only figure would "
              f"never show this)")
    else:
        print("  VRAM released after peak: not measured (NVML unavailable)")
    print(json.dumps({k: v for k, v in meta.items() if k != "passes"}, indent=None))
    return 0


if __name__ == "__main__":
    from _bench_guard import BenchGuard

    # `--compare` only reads two .pt files off disk and touches no GPU, so it does not take the
    # lock. Everything else here loads a diffusion model and samples it, and this file is the
    # single largest GPU consumer in tools/ -- 757 lines, two models, minutes of sampling -- and
    # it took no lock at all. That is worse for the sibling session than for this one: with no
    # lock file present their `Assert-GpuLock` would have been granted while this was running,
    # which is exactly the "20.49 GiB resident with the lock file absent" case `_bench_guard.py`
    # was written for.
    if "--compare" in sys.argv:
        raise SystemExit(main())
    with BenchGuard("comfy_portable:nunchaku_compare") as _guard:
        if _guard.refused:
            print(_guard.refused)
            raise SystemExit(1)
        raise SystemExit(main())
