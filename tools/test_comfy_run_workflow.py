"""Regression tests for `comfy_run_workflow.py`'s typed API-format model (ticket 05).

No pytest in this embedded interpreter -- verified by execution, see CLAUDE.md -- so this file
carries its own runner: `test_*` functions, bare asserts, a `main()` that calls them, PASS/FAIL
per case, `sys.exit(1)` on any failure, and a block at the end naming what this run did not cover.

    python_embeded\\python.exe -s tools/test_comfy_run_workflow.py

WHAT THIS FILE PROVES (the ticket's closing criterion, checked here, not just read):
  - `Wire`, `Value`, `Node` exist as types (`test_node_model_shape`).
  - the seed override in `main()` is `Node.set_widget`, a model method, not `node["inputs"][key]
    = ...` mutation from inside `main()` (`test_main_uses_set_widget_not_dict_mutation`).
  - no `isinstance(..., list)` survives anywhere as the test for "is this input a wire"
    (`test_no_isinstance_list_wire_test_survives`) -- `Node.is_wire` answers from the `wired` set
    built at construction time instead, proven directly by
    `test_is_wire_does_not_infer_from_python_type`, which hands a Node a list-shaped VALUE that is
    NOT in `wired` and checks `is_wire` still says no.

THE GOLDEN-DUMP TESTS (`test_golden_dump_no_seed`, `test_golden_dump_with_seed`) are the actual
proof the refactor preserved behaviour, per the task's instructions: `--dump-api` converts a real
saved workflow to API format and exits without touching a server, so it is the one seam that can
be compared byte-for-byte across a refactor. The golden files under `tools/fixtures/` were
produced by running the PRE-refactor `comfy_run_workflow.py` (plain-dict prompt, `isinstance`
seed-override) against `tools/fixtures/object_info_ltx25.json` and
`ComfyUI/user/default/workflows/LTX25-int8-acceptance-v2.json`, saved with `--dump-api`, then
diffed byte-for-byte (`diff`, not `assert` -- a real repo diff, printed clean) against the same
invocation on the POST-refactor code before these goldens were committed as the regression check.
This suite re-runs that same conversion in-process (importing `ui_to_api` and `prompt_to_api`
directly, matching how `main()` calls them) and asserts the JSON text is identical, so a future
change that silently reintroduces different behaviour breaks a test instead of needing a human to
notice a diff.

`tools/fixtures/object_info_ltx25.json` was produced by EXECUTING ComfyUI's own node introspection
-- importing `nodes` and `comfy_extras/nodes_lt*.py` / `nodes_custom_sampler.py` (via
`nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)`, so the 60 packages under
`custom_nodes/` are never touched) and calling the exact `node_info()` function `server.py`'s
`/object_info` route calls -- not hand-transcribed from reading the node source. No CUDA
allocation happens in that path; it is Python class/schema introspection. See the (deleted after
use) generator script referenced in this ticket's `## Resolucao` section for the exact code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent

_spec = importlib.util.spec_from_file_location("comfy_run_workflow", TOOLS / "comfy_run_workflow.py")
crw = importlib.util.module_from_spec(_spec)
sys.modules["comfy_run_workflow"] = crw  # dataclasses' ClassVar/InitVar string-annotation
# resolution (from `from __future__ import annotations`) looks the module up by name here;
# without this line, exec_module dies inside `dataclasses._is_type` on Python 3.13.
_spec.loader.exec_module(crw)

FIXTURES = TOOLS / "fixtures"
WORKFLOW = ROOT / "ComfyUI" / "user" / "default" / "workflows" / "LTX25-int8-acceptance-v2.json"
OBJECT_INFO = FIXTURES / "object_info_ltx25.json"


# --------------------------------------------------------------------------------------------
# The model itself
# --------------------------------------------------------------------------------------------

def test_node_model_shape():
    """`Wire`, `Value`, `Node` exist as the ticket specifies, and `Node` carries class_type +
    inputs (+ the `wired` bookkeeping this implementation adds to make is_wire answerable without
    inspecting a value's Python type)."""
    assert crw.Wire == tuple[str, int]
    assert crw.Value == str | int | float | bool

    node = crw.Node(class_type="KSamplerSelect", inputs={"sampler_name": "euler"})
    assert node.class_type == "KSamplerSelect"
    assert node.inputs == {"sampler_name": "euler"}
    assert node.wired == set()  # default_factory, not a shared mutable default


def test_set_widget_writes_a_widget_and_reports_it():
    node = crw.Node(class_type="RandomNoise", inputs={"noise_seed": 0}, wired=set())
    assert node.set_widget("noise_seed", 999) is True
    assert node.inputs["noise_seed"] == 999


def test_set_widget_is_a_noop_on_a_wired_input():
    """The core of the model: overriding a wired input must not clobber the wire. This is the
    exact case the old code guarded with `not isinstance(node["inputs"][key], list)`."""
    node = crw.Node(
        class_type="SamplerCustomAdvanced",
        inputs={"noise_seed": ("14", 0)},
        wired={"noise_seed"},
    )
    assert node.set_widget("noise_seed", 999) is False
    assert node.inputs["noise_seed"] == ("14", 0), "a no-op must not touch the wire"


def test_set_widget_is_a_noop_on_an_absent_key():
    node = crw.Node(class_type="KSamplerSelect", inputs={"sampler_name": "euler"})
    assert node.set_widget("seed", 5) is False
    assert "seed" not in node.inputs


def test_is_wire_does_not_infer_from_python_type():
    """The ticket's closing criterion in one assertion: a Node handed a LIST-shaped literal
    VALUE that was never marked as wired must still say `is_wire` is False. If `is_wire` (or
    `set_widget`) were reimplemented as `isinstance(value, list)` -- the exact regression this
    ticket exists to prevent -- this test would flip to True and fail."""
    node = crw.Node(class_type="SaveImage", inputs={"filename_prefix": ["not", "a", "wire"]})
    assert node.wired == set()
    assert node.is_wire("filename_prefix") is False
    # And set_widget must therefore be willing to overwrite it -- proving the behaviour, not
    # just the boolean.
    assert node.set_widget("filename_prefix", "ok") is True
    assert node.inputs["filename_prefix"] == "ok"


def test_to_api_shape_and_json_round_trip():
    node = crw.Node(
        class_type="CLIPTextEncode",
        inputs={"text": "hello", "clip": ("2", 0)},
        wired={"clip"},
    )
    api = node.to_api()
    assert list(api.keys()) == ["class_type", "inputs"], "server-facing key order matters"
    assert api == {"class_type": "CLIPTextEncode", "inputs": {"text": "hello", "clip": ("2", 0)}}
    # A tuple Wire must serialize exactly like the list ComfyUI's own JSON uses -- json.dumps
    # does not distinguish tuple from list, but this pins that down explicitly rather than
    # trusting it.
    assert json.loads(json.dumps(api))["inputs"]["clip"] == ["2", 0]


def test_prompt_to_api_maps_every_node():
    prompt = {
        "1": crw.Node(class_type="KSamplerSelect", inputs={"sampler_name": "euler"}),
        "2": crw.Node(class_type="VAEDecode", inputs={"samples": ("1", 0), "vae": ("3", 0)},
                      wired={"samples", "vae"}),
    }
    api = crw.prompt_to_api(prompt)
    assert set(api) == {"1", "2"}
    assert api["2"]["inputs"]["vae"] == ("3", 0)


# --------------------------------------------------------------------------------------------
# The symptoms the ticket names -- checked against the actual file, not just against the model
# --------------------------------------------------------------------------------------------

def test_main_uses_set_widget_not_dict_mutation():
    """`main()`'s seed override must be `node.set_widget(...)`, and must NOT contain the old
    `node["inputs"][key] = ...` mutation or `node["inputs"]`/`node["class_type"]` subscripting at
    all -- `prompt` holds `Node` objects now, not bare dicts. (main() still legitimately does
    `isinstance(m, (list, tuple))` elsewhere, on the /history status-message log -- an unrelated
    question -- so this checks for the specific old pattern by substring, not for isinstance's
    absence outright; the file-wide sweep below covers the wire-test question precisely.) Reads
    the shipped source of `main`, not a copy -- a regression that reintroduces the old pattern
    fails this test even if every other test above still passes with the model unused."""
    import inspect
    src = inspect.getsource(crw.main)
    assert "set_widget" in src, "main() must call Node.set_widget for the seed override"
    assert 'node["inputs"]' not in src, "main() must not subscript a Node's inputs directly"
    assert "node['class_type']" not in src and 'node["class_type"]' not in src, (
        "main() must use node.class_type, not dict-style subscripting")


def test_no_isinstance_list_wire_test_survives_in_the_file():
    """File-wide sweep for the specific pattern the ticket calls out: `isinstance(<something>,
    list)` used to ask "is this input a wire". The other `isinstance(x, (list, tuple))` calls in
    this file test whether raw JSON *shapes* (a link tuple, an INPUT_TYPES spec) look like an
    array before indexing them -- a different question, about parsing, not about wire-ness -- and
    are left alone; only the exact single-type `isinstance(v, list)` shape is checked for."""
    import re
    text = (TOOLS / "comfy_run_workflow.py").read_text(encoding="utf-8")
    offenders = []
    for lineno, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0]  # strip a trailing comment; none of these lines put a
        # literal '#' inside a string, so this is a safe enough split for a self-scan
        for m in re.finditer(r"isinstance\(\s*([^,]+?)\s*,\s*list\s*\)", code):
            var = m.group(1)
            if var == "typ":
                # `isinstance(typ, list)`, x2: is this INPUT_TYPES spec a combo (list of
                # options)? A question about a type SPEC, not about whether an input VALUE is a
                # wire -- the question this ticket is about. Left alone deliberately.
                continue
            offenders.append((lineno, line.strip()))
    assert offenders == [], (
        f"found isinstance(x, list) used as something other than the typ-spec check: {offenders}")


# --------------------------------------------------------------------------------------------
# Golden dump: byte-identical --dump-api output, proving the refactor preserved behaviour
# --------------------------------------------------------------------------------------------

def _convert(seed: int | None) -> str:
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    object_info = json.loads(OBJECT_INFO.read_text(encoding="utf-8"))
    prompt, _warnings = crw.ui_to_api(wf, object_info)
    if seed is not None:
        for _nid, node in prompt.items():
            for key in ("noise_seed", "seed"):
                node.set_widget(key, seed)
    return json.dumps(crw.prompt_to_api(prompt), indent=2)


def test_golden_dump_no_seed():
    assert WORKFLOW.is_file(), f"missing fixture workflow {WORKFLOW}"
    got = _convert(seed=None)
    golden = (FIXTURES / "golden_no_seed.json").read_text(encoding="utf-8")
    assert got == golden, "conversion output changed vs the pre-refactor golden (no --seed)"


def test_golden_dump_with_seed():
    got = _convert(seed=12345)
    golden = (FIXTURES / "golden_seed_12345.json").read_text(encoding="utf-8")
    assert got == golden, "conversion output changed vs the pre-refactor golden (--seed 12345)"


def test_golden_dump_covers_a_real_wired_and_a_real_wire_free_seed_case():
    """Sanity check on the golden itself, so a golden file quietly emptied of signal (e.g.
    someone swaps in a workflow with no seed node) does not still pass the two tests above.
    Confirms node 14 (RandomNoise.noise_seed) took the override and at least one wire survived
    conversion as a 2-element array, in the actual golden file on disk."""
    golden = json.loads((FIXTURES / "golden_seed_12345.json").read_text(encoding="utf-8"))
    assert golden["14"]["inputs"]["noise_seed"] == 12345
    wires = [v for node in golden.values() for v in node["inputs"].values()
             if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
    assert len(wires) >= 5, "golden fixture should contain several real wires to be meaningful"


# NOT a test, deliberately not named test_*, printed by the runner instead.
NOT_COVERED = (
    "This suite does not talk to a live ComfyUI server: it exercises ui_to_api/prompt_to_api/"
    "Node in-process and via --dump-api only. main()'s HTTP path (/prompt POST, /history poll, "
    "the cache-hit-vs-render heuristic around line ~344, the warning-string severity question "
    "from ticket 03, and main() decomposition from ticket 06) is untouched here by design -- "
    "those are other tickets' scope, not because this run verified them. "
    "tools/fixtures/object_info_ltx25.json reflects ComfyUI 0.33.0's schema for exactly the 16 "
    "node classes LTX25-int8-acceptance-v2.json uses; it will silently go stale if those nodes' "
    "INPUT_TYPES change upstream and nobody regenerates it -- this suite cannot detect that "
    "without a live server to compare against."
)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a crash is still a reportable failure
            failures += 1
            print(f"FAIL  {name}: unexpected {exc!r}")

    print(f"\nNAO COBERTO: {NOT_COVERED}")
    raise SystemExit(1 if failures else 0)
