"""Regression tests for `comfy_run_workflow.py`'s typed API-format model (ticket 05) and its
typed conversion notes (ticket 03: unknown class / >1 widget misalignment are `Note(level=
"fatal")`, and `main()` refuses to submit while one is outstanding, unless `--force`).

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
    # Ticket 06 moved the seed loop out of main() into apply_seed(). The assertion is unchanged --
    # the override goes through Node.set_widget and never subscripts a Node -- only its address
    # did. Both are read, because main() must not grow its own copy back.
    src = inspect.getsource(crw.apply_seed) + inspect.getsource(crw.main)
    assert "set_widget" in src, "the seed override must go through Node.set_widget"
    assert 'node["inputs"]' not in src, "the seed override must not subscript a Node's inputs"
    assert "node['class_type']" not in src and 'node["class_type"]' not in src, (
        "use node.class_type, not dict-style subscripting")


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
# Note (ticket 03): unknown class and >1 widget misalignment are fatal; fatal blocks main()
# --------------------------------------------------------------------------------------------

def test_note_shape():
    n = crw.Note(level="fatal", node="7", text="boom")
    assert n.level == "fatal"
    assert n.node == "7"
    assert n.text == "boom"


def test_unknown_class_is_fatal():
    """The exact symptom this ticket is about: the node vanishes from `prompt`, AND the note
    describing that is level='fatal', not indistinguishable free text."""
    wf = {"nodes": [{"id": 1, "type": "TotallyUnknownNodeXYZ", "mode": 0, "inputs": [],
                      "widgets_values": []}], "links": []}
    prompt, notes = crw.ui_to_api(wf, object_info={})
    assert "1" not in prompt, "the unknown node must still vanish from the prompt"
    fatal = [n for n in notes if n.level == "fatal"]
    assert len(fatal) == 1
    assert fatal[0].node == "1"
    assert "does not know class" in fatal[0].text


def test_muted_node_is_info_not_fatal():
    wf = {"nodes": [{"id": 1, "type": "Anything", "mode": 2, "inputs": [],
                      "widgets_values": []}], "links": []}
    _prompt, notes = crw.ui_to_api(wf, object_info={"Anything": {"input": {}}})
    assert len(notes) == 1
    assert notes[0].level == "info"


def _defn(names: list[str]) -> dict:
    """A minimal /object_info entry with `names` as required scalar (STRING) widgets, in order."""
    return {
        "input": {"required": {n: ["STRING", {}] for n in names}},
        "input_order": {"required": names},
    }


def test_widget_off_by_one_missing_is_info():
    """CLIPLoader's real shape (this file's own regression fixture): one trailing widget value
    absent must stay level='info' and must NOT be treated as a misalignment -- this is the exact
    case the orchestrator's ticket instructions call out by name."""
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [],
                      "widgets_values": ["a", "b"]}], "links": []}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": _defn(["w1", "w2", "w3"])})
    assert len(notes) == 1
    assert notes[0].level == "info"
    assert "trailing ones left at server default" in notes[0].text


def test_widget_off_by_two_missing_is_fatal():
    """More than one widget missing is the ticket's 'desalinhamento de mais de um widget' --
    must be fatal, not the same info-level note as off-by-one."""
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [], "widgets_values": ["a"]}],
          "links": []}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": _defn(["w1", "w2", "w3"])})
    assert len(notes) == 1
    assert notes[0].level == "fatal"


def test_widget_extra_by_two_is_fatal():
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [],
                      "widgets_values": ["a", "b", "c", "d"]}], "links": []}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": _defn(["w1", "w2"])})
    assert len(notes) == 1
    assert notes[0].level == "fatal"


def test_widget_extra_by_one_explained_by_a_companion_is_silent():
    """A widget the SERVER marks as having a UI companion eats two slots in widgets_values.

    `control_after_generate` (seed) and `image_upload` (LoadImage) are the two markers
    /object_info publishes for "the frontend draws this as two controls". When the extra
    value is accounted for by one of them, the count lines up exactly and nothing is said.
    """
    defn = _defn(["w1", "w2"])
    defn["input"]["required"]["w1"] = ["INT", {"control_after_generate": True}]
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [],
                      "widgets_values": [7, "randomize", "c"]}], "links": []}
    prompt, notes = crw.ui_to_api(wf, object_info={"N": defn})
    assert notes == []
    # e o essencial: o companheiro foi consumido NA POSICAO dele, nao no fim da lista
    assert prompt["2"].inputs == {"w1": 7, "w2": "c"}


def test_widget_extra_by_one_NOT_explained_is_fatal():
    """CONTRATO MUDADO EM 2026-09-12, de proposito.

    Este teste dizia antes que extra-de-um e sempre mudo, "a forma normal do
    control_after_generate de um node de seed" -- so que o fixture nao tem seed nenhum, e
    era essa aproximacao que segurava o bug: `ui_to_api` aceitava o extra como normal e
    depois fazia `zip(names, vals)`, pareando desde o indice 0, entao TUDO depois do extra
    deslizava uma casa. Num KSampler de verdade isso dava steps='randomize', cfg=10,
    sampler_name=1.0, denoise='simple' -- um grafo que pode executar e devolver resultado
    plausivel e errado, que e a falha que este arquivo inteiro existe para impedir.

    Agora o companheiro e contado pelo marcador do servidor (teste acima). Um extra que
    NENHUM marcador explica nao tem hipotese benigna sobrando: e desalinhamento.
    """
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [],
                      "widgets_values": ["a", "b", "c"]}], "links": []}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": _defn(["w1", "w2"])})
    assert len(notes) == 1
    assert notes[0].level == "fatal"


def test_wired_widget_still_consumes_its_slot_in_widgets_values():
    """`widgets_values` e indexado pela lista COMPLETA de widgets, inclusive os que viraram
    entrada ligada por fio -- o frontend guarda o valor velho na posicao.

    Regressao real: um EmptySD3LatentImage com width/height ligados a um ResolutionSelector
    chega com vals [1024, 1024, 1] e names ['batch_size'], e a leitura so-dos-nao-ligados
    atribuia batch_size=1024. Passava na validacao do servidor e gerava um lote de 1024.
    """
    wf = {"nodes": [
        {"id": 1, "type": "SRC", "mode": 0, "inputs": [], "outputs": [{"links": [9]}]},
        {"id": 2, "type": "N", "mode": 0, "widgets_values": [1024, 1024, 1],
         "inputs": [{"name": "w1", "type": "INT", "link": 9}]},
    ], "links": [[9, 1, 0, 2, 0, "INT"]]}
    defn = {"input": {"required": {"w1": ["INT", {}], "w2": ["INT", {}], "w3": ["INT", {}]}},
            "input_order": {"required": ["w1", "w2", "w3"]}}
    prompt, _notes = crw.ui_to_api(wf, object_info={"N": defn, "SRC": _defn([])})
    assert prompt["2"].inputs["w2"] == 1024
    assert prompt["2"].inputs["w3"] == 1, "o valor do widget ligado nao pode escorregar"


def test_orphan_wire_into_optional_input_is_dropped_and_named():
    """Um node mutado sai do prompt; quem apontava para ele fica com a referencia pendurada.

    O servidor responde 400 com uma mensagem que so cita o ID que falta, em TODOS os nodes
    que o referenciavam menos o proprio. Foi assim que o workflow oficial do Krea2 Edit --
    que muta a segunda referencia de proposito -- deu `'90'` em tres nodes, nenhum deles o 90.
    """
    wf = {"nodes": [
        {"id": 1, "type": "SRC", "mode": 4, "inputs": [], "outputs": [{"links": [9]}]},
        {"id": 2, "type": "N", "mode": 0, "widgets_values": [],
         "inputs": [{"name": "opc", "type": "IMAGE", "link": 9}]},
    ], "links": [[9, 1, 0, 2, 0, "IMAGE"]]}
    defn = {"input": {"required": {}, "optional": {"opc": ["IMAGE", {}]}},
            "input_order": {"required": [], "optional": ["opc"]}}
    prompt, notes = crw.ui_to_api(wf, object_info={"N": defn, "SRC": _defn([])})
    assert "opc" not in prompt["2"].inputs
    assert any(n.level == "info" and "opc" in n.text and "1" in n.text for n in notes)
    assert not any(n.level == "fatal" for n in notes)


def test_orphan_wire_into_required_input_is_fatal():
    """Mesma situacao numa entrada OBRIGATORIA: nao ha o que inventar, o grafo esta
    incompleto, e deixar passar seria submeter um prompt mutilado."""
    wf = {"nodes": [
        {"id": 1, "type": "SRC", "mode": 4, "inputs": [], "outputs": [{"links": [9]}]},
        {"id": 2, "type": "N", "mode": 0, "widgets_values": [],
         "inputs": [{"name": "obg", "type": "IMAGE", "link": 9}]},
    ], "links": [[9, 1, 0, 2, 0, "IMAGE"]]}
    defn = {"input": {"required": {"obg": ["IMAGE", {}]}},
            "input_order": {"required": ["obg"]}}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": defn, "SRC": _defn([])})
    assert any(n.level == "fatal" and "obg" in n.text for n in notes)


def test_ui_only_node_is_info_not_fatal():
    """`Note` e `MarkdownNote` nao tem classe no servidor POR DESENHO. Trata-las como classe
    desconhecida recusava todo workflow com um post-it no canvas -- inclusive o oficial do
    Krea2 Edit, que traz oito. E um fatal sobre o qual nao ha nada a fazer no grafo."""
    wf = {"nodes": [{"id": 3, "type": "Note", "mode": 0, "inputs": [],
                      "widgets_values": ["um lembrete"]}], "links": []}
    prompt, notes = crw.ui_to_api(wf, object_info={})
    assert prompt == {}
    assert len(notes) == 1
    assert notes[0].level == "info"


def test_required_input_filled_from_default_is_warn():
    wf = {"nodes": [{"id": 2, "type": "N", "mode": 0, "inputs": [], "widgets_values": []}],
          "links": []}
    # A required input of a non-widget type (a socket, not INT/FLOAT/STRING/...) with a
    # 'default' in its opts -- isolates the required-fill path from the widget-count path,
    # which is exercised separately above.
    defn = {"input": {"required": {"w1": ["MODEL", {"default": "hi"}]}}, "input_order": {}}
    _prompt, notes = crw.ui_to_api(wf, object_info={"N": defn})
    assert len(notes) == 1
    assert notes[0].level == "warn"
    assert "filled with the server default" in notes[0].text


def test_main_refuses_fatal_without_force_and_does_not_touch_network():
    """The ticket's actual closing criterion, exercised on `main()` itself, not just on
    `ui_to_api`: a fatal Note must block submission before any HTTP call. Uses
    --object-info-file (offline, skips wait_for_server) and omits --dump-api, so if this test
    reached the /prompt POST it would try to open a real socket to 127.0.0.1:8190 -- exactly the
    live-network, live-server action forbidden in this environment. Reaching return 4 without an
    exception is the proof that never happens when a fatal Note is outstanding and --force was
    not given."""
    import tempfile
    wf = {"nodes": [{"id": 1, "type": "TotallyUnknownNodeXYZ", "mode": 0, "inputs": [],
                      "widgets_values": []}], "links": []}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wf_path = tdp / "wf.json"
        oi_path = tdp / "oi.json"
        wf_path.write_text(json.dumps(wf), encoding="utf-8")
        oi_path.write_text(json.dumps({}), encoding="utf-8")

        saved_argv = sys.argv
        sys.argv = ["comfy_run_workflow.py", "--workflow", str(wf_path),
                    "--object-info-file", str(oi_path)]
        try:
            rc = crw.main()
        finally:
            sys.argv = saved_argv
        assert rc == 4, f"expected refusal exit code 4, got {rc}"


def test_main_reads_a_workflow_that_carries_a_utf8_bom():
    """MEDIDO 2026-09-01: um dos 79 workflows reais do dono comeca com `ef bb bf` --
    `SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_safe_720p_Q8.json`. Com `encoding="utf-8"` o `json.loads`
    morre em `Unexpected UTF-8 BOM` antes de olhar um unico no, entao o arquivo era simplesmente
    inabrivel por esta ferramenta. Editor do Windows grava BOM sem perguntar, entao isto reaparece.

    O par de casos e o ponto: o MESMO workflow, com e sem BOM, tem de dar o MESMO codigo de saida.
    Testar so o caso com BOM provaria que ele nao explode, nao que `utf-8-sig` deixou o caso comum
    intacto -- e trocar codec e exatamente o tipo de conserto que quebra o outro lado em silencio.
    """
    import tempfile
    wf = {"nodes": [{"id": 1, "type": "TotallyUnknownNodeXYZ", "mode": 0, "inputs": [],
                     "widgets_values": []}], "links": []}
    saidas = {}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        oi_path = tdp / "oi.json"
        oi_path.write_text(json.dumps({}), encoding="utf-8")
        for rotulo, codec in (("sem_bom", "utf-8"), ("com_bom", "utf-8-sig")):
            wf_path = tdp / f"wf_{rotulo}.json"
            wf_path.write_text(json.dumps(wf), encoding=codec)
            cru = wf_path.read_bytes()
            assert (cru[:3] == b"\xef\xbb\xbf") == (rotulo == "com_bom"), \
                f"{rotulo}: o fixture nao tem o BOM que deveria ter, primeiros bytes {cru[:3]!r}"
            saved_argv = sys.argv
            sys.argv = ["comfy_run_workflow.py", "--workflow", str(wf_path),
                        "--object-info-file", str(oi_path)]
            try:
                saidas[rotulo] = crw.main()
            finally:
                sys.argv = saved_argv
    assert saidas["com_bom"] == saidas["sem_bom"] == 4, (
        f"BOM mudou o resultado: {saidas}. Com utf-8 puro o caso com_bom levantava "
        f"JSONDecodeError em vez de devolver 4.")


# --------------------------------------------------------------------------------------------
# Ticket 04: cache detection uses the server's own duration, not client wall, and a cache hit
# gets its own exit code. The /prompt + /history path needs a live server (see NOT_COVERED),
# so this is a static proof against the shipped source -- same technique as the isinstance
# sweep above -- not a behavioural run of main() past the queue POST.
# --------------------------------------------------------------------------------------------

def test_cache_hit_threshold_is_a_named_module_constant():
    assert crw.CACHE_HIT_THRESHOLD_S == 5.0
    assert isinstance(crw.CACHE_HIT_THRESHOLD_S, float)


def test_cache_detection_uses_server_side_duration_not_wall():
    """The exact regression this ticket is about: `if wall < 5.0 and files:` must be gone from
    the module, and the cache_hit expression must be built from `server_side_s` (the server's
    own execution_start -> execution_success span), not from `wall` (this client's estimate).

    Ticket 06 moved this computation out of `main()` into `Entry.cache_hit` (a property, one
    place the whole module computes it from, instead of main() recomputing it inline every
    time it prints something). The proof this test makes is unchanged -- only WHERE it looks
    changed, because the code it is proving something about moved. Before ticket 06 this read
    `inspect.getsource(crw.main)`; a `cache_hit = ...` assignment doesn't exist in main() any
    more (main() now reads `entry.cache_hit`), so this reads the property itself."""
    import inspect
    import re
    full_src = (TOOLS / "comfy_run_workflow.py").read_text(encoding="utf-8")
    assert "wall < 5.0" not in full_src, "the old wall-based heuristic must not survive"
    assert "wall < CACHE_HIT_THRESHOLD_S" not in full_src, (
        "the threshold moved to server_side_s, not merely renamed on wall")
    assert "server_side_s" in full_src, "the module must compute a first-class server-side duration"

    prop_src = inspect.getsource(crw.Entry.cache_hit.fget)
    assert "server_side_s" in prop_src and "CACHE_HIT_THRESHOLD_S" in prop_src
    # The docstring is allowed to say "wall" in English prose (it explains why wall is NOT
    # used); what must never say `wall` is the actual `return` expression -- same rationale
    # ticket 06's original test used to isolate the assignment's RHS instead of scanning the
    # whole function body.
    m = re.search(r"return\s+(.+)", prop_src, re.DOTALL)
    assert m, "expected a `return ...` expression in Entry.cache_hit"
    rhs = m.group(1)
    assert re.search(r"\bwall\b", rhs) is None, (
        f"cache_hit must be computed from server_side_s, not wall -- got: {rhs}")


def test_cache_hit_gets_its_own_exit_code():
    """A cache hit must not be folded into exit 0 (success) or exit 1 (failure) -- it needs a
    code a caller (run_e2e_comfy.ps1) can distinguish from both.

    Ticket 06: main() now reads `entry.cache_hit` (a property) rather than a local `cache_hit`
    variable, since the cache-hit computation itself moved to `Entry.cache_hit` -- the pattern
    below matches that new shape; the assertion it proves (a cache hit gets exit code 5, on its
    own, not folded into 0 or 1) is unchanged."""
    import inspect
    import re
    src = inspect.getsource(crw.main)
    assert re.search(r"if\s+entry\.cache_hit\s*:\s*\n\s*return\s+5", src), (
        "expected `if entry.cache_hit: return 5` (its own exit code) in main()")


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
    "Node/Note in-process, and main() only up to the point where a fatal Note returns 4 -- never "
    "past it. Ticket 06 split main() into Comfy (the HTTP boundary: wait_up/object_info/submit/"
    "history/queue), run_and_wait (submit+poll -> Entry) and report (prints an Entry) -- those "
    "units now EXIST and are read directly by the static tests below, but none of them is called "
    "here with a real or fake server: the --force-allows-submission path, the actual /prompt "
    "POST, and the /history poll loop are untouched here by design (that HTTP path needs a live "
    "server, which this environment does not permit) -- those are other tickets' scope or another "
    "run's job, not verified by this one. Ticket 04's cache-vs-render check is covered only "
    "STATICALLY here (source-pattern tests on Entry.cache_hit's and main()'s shipped text: "
    "server_side_s / CACHE_HIT_THRESHOLD_S / 'return 5' exist, 'wall < 5.0' is gone, and the "
    "cache_hit expression does not name wall) -- whether a real /history response from a "
    "cache-hit prompt drives server_side_s under the threshold and main() actually exits 5 "
    "against a live server was NOT run here. "
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
