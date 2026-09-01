"""The write contract on `svdq_to_bf16.write_checkpoint`, re-runnably.

WHY THIS FILE EXISTS, separately from `test_svdq_verify.py`. That file's `__main__` runs a GPU
battery, so it cannot be the home for a check that must be runnable while a sibling session holds
the card -- and this check is precisely the kind you want to run *then*, because it is about a
failure that produces a plausible file rather than an exception.

`svdq_to_bf16.py` was the only writer in `tools/` without the contract the other six share. It
opened its partial `"wb"` (clobbering a crashed run's leftover), never compared bytes written
against bytes planned, never `fsync`ed, and had no `try/finally`. That gap is worse here than in
any of the six for two reasons the converter's own docstring gives:

  - a safetensors header is self-describing, so a file short by one tensor still parses and still
    LOADS. The tail tensors come back as garbage and the model produces a wrong image, not an
    error.
  - this is the one converter whose output has no BF16 original to fall back on. Recovering it is
    the entire reason the tool exists.

The contract was added on 2026-08-22 and its docstring records that it was executed once, by hand,
on a temp-dir fixture. **Executed once by hand is not coverage** -- nothing re-runs it, and
`--limit` is no longer a cheap route to the writer either. So: this file, which needs no GPU, no
CUDA, and no multi-GiB source.

    .\\python_embeded\\python.exe -s .\\tools\\test_svdq_write_contract.py

Exits non-zero if anything fails.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ComfyUI"))

import torch  # noqa: E402

import svdq_to_bf16 as sv  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def rule(title: str) -> None:
    print()
    print("-" * 76)
    print(title)
    print("-" * 76)


def build_source(path: Path, tensors: dict[str, torch.Tensor]) -> tuple[int, dict]:
    """A real safetensors file, hand-assembled. Returns (data_start, header)."""
    header: dict = {}
    offset = 0
    blobs = []
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "F32" if tensor.dtype is torch.float32 else "I8",
                        "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for raw in blobs:
            handle.write(raw)
    return 8 + len(payload), header


def plan(src_header: dict, produced: dict[str, torch.Tensor],
         copied: list[str]) -> tuple[list, bytes, int]:
    """The (entries, header blob, planned bytes) triple `write_checkpoint` takes.

    Built here rather than imported so the test drives the writer directly and does not depend on
    the SVDQuant recovery path, which needs a real fused checkpoint.
    """
    entries: list = []
    out_header: dict = {}
    offset = 0
    for name in copied:
        info = src_header[name]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        out_header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [offset, offset + size]}
        entries.append((name, None, ("copy", info)))
        offset += size
    for name, tensor in produced.items():
        raw_len = tensor.numel() * tensor.element_size()
        out_header[name] = {"dtype": "F32", "shape": list(tensor.shape),
                            "data_offsets": [offset, offset + raw_len]}
        entries.append((name, None, ("write", tensor)))
        offset += raw_len
    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    return entries, blob, offset


def fixture(root: Path):
    torch.manual_seed(7)
    tensors = {"kept.a": torch.arange(4, dtype=torch.float32),
               "kept.b": torch.arange(3, dtype=torch.float32) * 10}
    src = root / "src.safetensors"
    data_start, header = build_source(src, tensors)
    produced = {"recovered.w": torch.arange(5, dtype=torch.float32) * 100}
    entries, blob, planned = plan(header, produced, ["kept.a", "kept.b"])
    return src, data_start, entries, blob, planned, tensors, produced


# ---------------------------------------------------------------------------------------------

def test_an_honest_write_round_trips(root: Path) -> None:
    src, data_start, entries, blob, planned, tensors, produced = fixture(root)
    out, partial = root / "out.safetensors", root / "out.safetensors.partial"

    sv.write_checkpoint(src, data_start, out, partial, entries, blob, planned)

    check("the output exists", out.is_file())
    check("the partial is gone", not partial.exists(),
          "finally: partial.unlink() on the success path too")

    from safetensors.torch import load_file
    got = load_file(str(out))
    check("every planned tensor is present", set(got) == set(tensors) | set(produced), sorted(got))
    for name, want in {**tensors, **produced}.items():
        same = torch.equal(got[name], want)
        check(f"  {name} round-trips byte-identical", same,
              "" if same else f"{got[name].tolist()} != {want.tolist()}")


def test_a_truncated_write_raises_and_leaves_nothing(root: Path) -> None:
    """The failure the contract exists for, and the reason it matters more here than elsewhere.

    A short safetensors still PARSES: the header describes offsets the body does not reach, and the
    tail comes back as garbage. Without `written == planned` this file would have been
    `os.replace`d into position and the tool would have printed `done:`.
    """
    src, data_start, entries, blob, planned, _, _ = fixture(root)
    out, partial = root / "short.safetensors", root / "short.safetensors.partial"

    raised = None
    try:
        # planned is inflated, so the honest write comes up short against it
        sv.write_checkpoint(src, data_start, out, partial, entries, blob, planned + 16)
    except RuntimeError as error:
        raised = error

    check("it raises RuntimeError", isinstance(raised, RuntimeError), repr(raised))
    check("and the message names both numbers",
          raised is not None and "length mismatch" in str(raised) and "planned" in str(raised),
          str(raised))
    check("no output file was left behind", not out.exists(),
          "" if not out.exists() else "A SHORT FILE WAS os.replace'd INTO POSITION")
    check("no partial was left behind", not partial.exists(),
          "" if not partial.exists() else "the next run would clobber it, which is the old bug")


def test_a_stale_partial_is_refused_not_clobbered(root: Path) -> None:
    """`"xb"`, not `"wb"`. A crashed run's leftover is evidence, not scratch space."""
    src, data_start, entries, blob, planned, _, _ = fixture(root)
    out, partial = root / "stale.safetensors", root / "stale.safetensors.partial"
    partial.write_bytes(b"leftover from a crashed run")

    raised = None
    try:
        sv.write_checkpoint(src, data_start, out, partial, entries, blob, planned)
    except FileExistsError as error:
        raised = error

    check("it raises FileExistsError", isinstance(raised, FileExistsError), repr(raised))
    check("no output was produced", not out.exists())
    # The `finally` removes the partial even though this run did not create it. That is the right
    # trade only because `main()` refuses a stale partial BEFORE any work starts, so reaching here
    # means somebody called the writer directly -- see the next test.
    check("the writer's finally cleaned up", not partial.exists(),
          "note: main() refuses earlier, so this path is direct-call only")


def test_main_refuses_a_stale_partial_before_doing_any_work(root: Path) -> None:
    """The guard that makes the previous test's cleanup acceptable.

    Checked by reading rather than by running -- `main()` parses args and then loads a real fused
    checkpoint, which needs CUDA and a multi-GiB source. TRACED, and the grep that finds it is
    below rather than a line number, because four line-number citations in this file's neighbours
    drifted within one session.
    """
    lines = (HERE / "svdq_to_bf16.py").read_text(encoding="utf-8").splitlines()

    def line_of(predicate) -> int:
        return next((i for i, line in enumerate(lines, 1) if predicate(line)), -1)

    # The CALL, not the definition. The first version of this test searched for
    # `write_checkpoint(src` and matched `def write_checkpoint(src: Path, ...)` four hundred lines
    # earlier, then reported the guard as coming AFTER the writer. The test was wrong and the code
    # was right -- which is the reason to make the needle unambiguous rather than to loosen the
    # assertion until it passes.
    #
    # Updated 2026-09-01, when `svdq_to_bf16` adopted `_conversion`. `main()` no longer calls
    # `write_checkpoint` at all: it builds core entries and calls `conv.commit()`, and the refusal
    # is `conv.refuse_unsafe(allow_quantized_source=True)` rather than a hand-written line. So the
    # ordering assertion now targets whichever writer main() actually uses, and the local
    # `write_checkpoint` -- kept only so the three scenarios above still have something to drive --
    # is no longer required to be called. Asserting the old shape would have meant asserting that
    # the migration had not happened.
    commit = line_of(lambda l: "conv.commit(" in l)
    refuse = line_of(lambda l: "conv.refuse_unsafe(" in l)
    guard = line_of(lambda l: "conv.guard(" in l)
    definition = line_of(lambda l: l.lstrip().startswith("def write_checkpoint("))
    legacy_call = line_of(lambda l: "write_checkpoint(" in l and not l.lstrip().startswith("def "))

    check("main() writes through the shared core", commit != -1, f"commit at line {commit}")
    check("and refuses before writing", -1 < refuse < commit,
          f"refuse at {refuse}, commit at {commit}")
    check("and guards before writing", -1 < guard < commit,
          f"guard at {guard}, commit at {commit}")
    check("the legacy writer is defined but no longer on the production path",
          definition != -1 and legacy_call == -1,
          f"def at line {definition}, stray call at line {legacy_call}")


def test_the_contract_is_still_the_same_shape_as_its_six_siblings(root: Path) -> None:
    """Every writer creates its `.partial` exclusively -- itself, or through the shared core.

    Not a line-number assertion: those rotted four times in one session here while sibling agents
    edited the same files. This asserts the GUARANTEE, which is what the docstring promises.

    The `or delegates` half was added on 2026-09-01, when `to_native.py` became the first writer
    to adopt `_conversion.py` and stopped carrying `"xb"` of its own. Asserting the literal in
    each file would have made this test block the very migration ticket 08 exists to do -- a check
    that punishes the fix teaches people to delete the check.

    It is not a weakening, and the same migration proved why the assertion has to stay: the core
    was opening `"wb"`, so it was **weaker** than all six writers it replaces, and this test is
    what caught it. So the core is held to the same literal.
    """
    siblings = ["quant_w4a4.py", "quant_w4a8.py", "quant_int8.py",
                "quant_w4a4_smooth.py", "quant_mixed.py", "to_native.py", "svdq_to_bf16.py"]
    core = (HERE / "_conversion.py").read_text(encoding="utf-8")
    check("the shared write core opens its partial exclusively", '"xb"' in core)
    missing = []
    for name in siblings:
        body = (HERE / name).read_text(encoding="utf-8")
        if '"xb"' in body:
            continue
        if "import _conversion" in body and '"xb"' in core:
            continue
        missing.append(name)
    check("every writer opens its partial exclusively, itself or via the core",
          missing == [], f"missing in {missing}")

    body = (HERE / "svdq_to_bf16.py").read_text(encoding="utf-8")
    for needle, label in (("os.fsync", "fsync before replace"),
                          ("os.replace(partial, out)", "atomic rename"),
                          ("finally:", "cleanup on every path"),
                          ("length mismatch", "written == planned")):
        check(f"  svdq_to_bf16 keeps: {label}", needle in body)


def main() -> int:
    print("=" * 76)
    print("svdq_to_bf16 write contract -- no GPU, no CUDA, no multi-GiB source")
    print("=" * 76)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    with tempfile.TemporaryDirectory(prefix="svdq_contract_") as raw:
        for test in tests:
            rule(test.__name__)
            root = Path(raw) / test.__name__
            root.mkdir()
            try:
                test(root)
            except Exception:
                FAIL.append(test.__name__)
                print(f"  [FAIL] {test.__name__} raised")
                traceback.print_exc()

    print()
    print("=" * 76)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    print()
    print("NOT COVERED BY THIS FILE:")
    print("  - the SVDQuant recovery itself. Every tensor here is hand-built; nothing checks that")
    print("    the dequantized weights are correct, only that what was planned is what landed.")
    print("  - a real checkpoint. The fixture is a few dozen bytes; behaviour at 11 GiB, where")
    print("    COPY_CHUNK actually loops and the source is on a network mount, is untested.")
    print("  - fsync's guarantee. The call is asserted to be there; whether the bytes survive a")
    print("    power cut is not something a test on this bench can show.")
    print("  - main()'s stale-partial refusal is TRACED by reading, not executed: reaching it")
    print("    needs CUDA and a real fused source.")
    print("=" * 76)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
