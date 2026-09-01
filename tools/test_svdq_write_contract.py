"""The write contract on `_conversion.Conversion.commit`, re-runnably.

REDIRECIONADO EM 2026-09-01. Este arquivo testava `svdq_to_bf16.write_checkpoint`, que era a copia
local do contrato naquela ferramenta. Os sete escritores desta bancada passaram a usar
`tools/_conversion.py`, entao os tres cenarios abaixo apontam para o nucleo e a funcao local foi
APAGADA -- 82 linhas, junto com `COPY_CHUNK` e `DTYPE_NAMES`, que so ela usava. O que se
testa e o mesmo: escrita honesta ida-e-volta, escrita curta, e `.partial` pre-existente.

**Um dos tres inverteu de proposito, e a inversao e o ponto.** O cenario do `.partial` pre-existente
afirmava que o `finally` do escritor limpava o arquivo de outro processo. O nucleo abre o `.partial`
FORA do `try` exatamente para que isso nao aconteca: se ele ja existe, o `FileExistsError` sobe sem
passar pelo `finally`, e o arquivo de quem esta escrevendo agora sobrevive. O teste agora afirma a
sobrevivencia. Nao e o teste que afrouxou; e o comportamento que ficou mais seguro.

O texto abaixo descreve por que este arquivo nasceu, e continua valendo.

---

The write contract on `svdq_to_bf16.write_checkpoint`, re-runnably.

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

import _conversion as C  # noqa: E402

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


def plan(src_header: dict, produced: dict[str, torch.Tensor], copied: list[str]) -> list:
    """As entradas do nucleo: faixas copiadas primeiro, tensores produzidos depois.

    Montadas aqui e nao importadas, para que o teste dirija o ESCRITOR diretamente e nao dependa do
    caminho de recuperacao do SVDQuant, que precisa de um checkpoint fundido de verdade.
    """
    entradas = [C.plan_copy(name, src_header[name]) for name in copied]
    entradas += [C.plan_write(name, tensor) for name, tensor in produced.items()]
    return entradas


def fixture(root: Path):
    torch.manual_seed(7)
    tensors = {"kept.a": torch.arange(4, dtype=torch.float32),
               "kept.b": torch.arange(3, dtype=torch.float32) * 10}
    src = root / "src.safetensors"
    _, header = build_source(src, tensors)
    produced = {"recovered.w": torch.arange(5, dtype=torch.float32) * 100}
    return src, header, plan(header, produced, ["kept.a", "kept.b"]), tensors, produced


# ---------------------------------------------------------------------------------------------

def test_an_honest_write_round_trips(root: Path) -> None:
    src, _, entries, tensors, produced = fixture(root)
    out = root / "out.safetensors"

    conv = C.Conversion(src, out)
    conv.commit(entries)
    partial = conv.partial

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
    src, header, entries, _, _ = fixture(root)
    out = root / "short.safetensors"

    # O nucleo calcula `planned` a partir das proprias entradas, entao nao da para inflar o total
    # por fora como a versao anterior fazia. A mesma falha se produz por dentro: uma entrada que
    # DECLARA mais bytes do que o produtor entrega. `plan_lazy` aceita `nbytes` explicito, que e o
    # que torna o plano calculavel sem GPU -- e tambem o que permite ele mentir.
    mentirosa = C.plan_lazy("recovered.w", "F32", [5], 20 + 16,
                            lambda: torch.arange(5, dtype=torch.float32) * 100)
    entries = entries[:-1] + [mentirosa]

    conv = C.Conversion(src, out)
    partial = conv.partial
    raised = None
    try:
        conv.commit(entries)
    except RuntimeError as error:
        raised = error

    check("it raises RuntimeError", isinstance(raised, RuntimeError), repr(raised))
    check("and the message names both numbers",
          raised is not None and "planned" in str(raised) and "36" in str(raised),
          str(raised))
    check("no output file was left behind", not out.exists(),
          "" if not out.exists() else "A SHORT FILE WAS os.replace'd INTO POSITION")
    check("no partial was left behind", not partial.exists(),
          "" if not partial.exists() else "the next run would clobber it, which is the old bug")


def test_a_stale_partial_is_refused_not_clobbered(root: Path) -> None:
    """`"xb"`, not `"wb"`. A crashed run's leftover is evidence, not scratch space."""
    src, _, entries, _, _ = fixture(root)
    out = root / "stale.safetensors"
    conv = C.Conversion(src, out)
    partial = conv.partial
    partial.write_bytes(b"leftover from a crashed run")

    raised = None
    try:
        conv.commit(entries)
    except FileExistsError as error:
        raised = error

    check("it raises FileExistsError", isinstance(raised, FileExistsError), repr(raised))
    check("no output was produced", not out.exists())
    # ESTA ASSERCAO INVERTEU EM 2026-09-01, e a inversao e o ponto. A versao anterior afirmava que
    # o `finally` do escritor apagava o `.partial` mesmo sem te-lo criado, e chamava isso de troca
    # aceitavel porque `main()` recusava antes. Mas o `.partial` que existe pode ser de um processo
    # ESCREVENDO AGORA, e apaga-lo e destruir trabalho alheio. O nucleo abre o `.partial` fora do
    # `try` justamente para que o FileExistsError nao passe pelo `finally`.
    check("the OTHER process's partial survives", partial.exists(),
          "aberto fora do try: o FileExistsError nao alcanca o finally")
    check("and it was not touched", partial.read_bytes() == b"leftover from a crashed run")
    partial.unlink()


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
    # Updated 2026-09-01, when `svdq_to_bf16` adopted `_conversion` e a copia local do contrato foi
    # APAGADA. `main()` monta entradas do nucleo e chama `conv.commit()`; a recusa e
    # `conv.refuse_unsafe(allow_quantized_source=True)` em vez de uma linha escrita a mao. Entao a
    # assercao de ordem agora aponta para o escritor que `main()` de fato usa, e passou a exigir
    # que NENHUM `write_checkpoint` sobreviva no arquivo -- a versao anterior desta linha aceitava
    # a funcao definida-mas-nao-chamada, que era o estado intermediario de meio dia.
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
    check("the local copy of the write contract is gone entirely",
          definition == -1 and legacy_call == -1,
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

    # As quatro metades restantes do contrato. Ate 2026-09-01 estas linhas procuravam os simbolos
    # DENTRO de `svdq_to_bf16.py`, que carregava a sua propria copia; agora a copia foi apagada e o
    # contrato vive no nucleo, entao e la que elas olham. A cobertura nao afrouxou -- ao contrario,
    # deixou de valer para um arquivo so e passou a valer para os sete de uma vez.
    for needle, label in (("os.fsync", "fsync before replace"),
                          ("os.replace(partial, self.output)", "atomic rename"),
                          ("finally:", "cleanup on every path"),
                          ("length mismatch", "written == planned")):
        check(f"  o nucleo mantem: {label}", needle in core, needle)


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
