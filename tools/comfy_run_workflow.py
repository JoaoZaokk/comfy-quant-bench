r"""Run a saved ComfyUI *UI-format* workflow against a running server, and time it.

(The `r` on this docstring is load-bearing: it holds Windows paths, and `\user`
in `ComfyUI\user\default\workflows` parses as a truncated `\uXXXX` escape, which
kills the file at import with a SyntaxError pointing at line 1.)

WHY THIS EXISTS. ComfyUI's `/prompt` endpoint takes API format
(`{node_id: {class_type, inputs}}`), but the files under
`ComfyUI/user/default/workflows/` are UI format (`nodes` + `links`) -- the
frontend converts them in the browser. Anything that wants to run a saved
workflow headlessly, with a stopwatch around it, has to do that conversion.
The alternative was to hand-write an API-format twin of the graph, which then
silently drifts from the file a human edits.

The conversion needs `/object_info` from the SAME server that will run the
prompt, because widget order is a property of the node definitions that server
has loaded -- a custom node pack updating its input list changes it. Reading a
stale `object_info.json` off disk is how a graph runs with parameters shifted
by one slot and still completes.

    .\python_embeded\python.exe -s .\tools\comfy_run_workflow.py ^
        --workflow "ComfyUI\user\default\workflows\LTX25-int8-acceptance-v2.json" ^
        --server 127.0.0.1:8190

WHAT THIS DOES NOT CHECK, printed again at the end of every run:
  - that the numbers it reports are a fair comparison against anything else;
    it times ONE server doing ONE prompt, storage and cache state included.
  - that the output is correct. It reports that files were written, not that
    they show what the prompt asked for. Look at them.
  - it cannot separate weight-load time from execution when the server is cold.
    Run it twice with a different `--seed` the second time (an identical prompt is answered
    by ComfyUI's node cache and exits 5), and compare the two server-side numbers.

API-FORMAT GRAPHS (review of 2026-09-29): `--api-prompt FILE` runs one API-format graph and
`--lista ordem.txt` runs one per line, skipping the UI conversion. `--saida x.jsonl` appends one
record per graph: grafo, prompt_id, status, erro, server_side_s, wall, cache_hit, files, and the
server's card/argv/versions from /system_stats. Exit 0 all rendered, 1 any failure/refusal/
timeout, 5 some cache hit. These modes go through `comfy_client.roda_lista`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# The HTTP client (Comfy, run_and_wait, Entry, bases_irmas, the cache-hit threshold) moved to
# `comfy_client.py` in the review of 2026-09-29 so that the battery scripts can import it instead of
# re-implementing it; it is re-exported here under the same names this module always had.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_client import (  # noqa: E402,F401  (re-exported)
    CACHE_HIT_THRESHOLD_S,
    Comfy,
    Entry,
    PollTimeout,
    PromptRefused,
    bases_irmas,
    codigo_de_saida,
    controles,
    grava_jsonl,
    http_get,
    http_post,
    le_lista,
    roda_lista,
    run_and_wait,
)

# The API-format prompt has exactly one central type: a node input is EITHER a literal value
# OR a wire to another node's output slot -- `["origin_id", slot]` in the JSON ComfyUI reads.
# This used to be nowhere in the code; every input was a bare dict value, and "is this a wire?"
# was answered ad hoc with `isinstance(v, list)` at each call site (main() had one, three levels
# into the orchestrator). Modeled here once so the question has one answer.
Wire = tuple[str, int]
Value = str | int | float | bool


@dataclass
class Node:
    """One node of a converted API-format prompt.

    `wired` names which `inputs` keys hold a `Wire` rather than a literal `Value` -- tracked
    explicitly at construction time (ui_to_api already knows this; it built the wire in the same
    loop), not re-derived later by inspecting the value's Python type. That is what lets
    `set_widget` answer "is this a wire?" without an isinstance check.
    """

    class_type: str
    inputs: dict[str, Value | Wire]
    wired: set[str] = field(default_factory=set)

    def is_wire(self, key: str) -> bool:
        return key in self.wired

    def set_widget(self, key: str, value: Value) -> bool:
        """Set a widget input to `value`. No-op if `key` is absent or is wired to another
        node's output -- overriding a wire's endpoints is not what a widget override means.
        Returns whether the value was actually set, so a caller can report what happened
        without re-asking "was this a wire" itself."""
        if key not in self.inputs or self.is_wire(key):
            return False
        self.inputs[key] = value
        return True

    def to_api(self) -> dict:
        """The `{class_type, inputs}` shape ComfyUI's /prompt endpoint expects. `wired` is
        bookkeeping for this module only -- the server has never heard of it."""
        return {"class_type": self.class_type, "inputs": self.inputs}


@dataclass
class Note:
    """One thing `ui_to_api` noticed about the conversion, typed instead of free text (ticket 03).

    Before this, every finding -- a muted node, an unknown node class, a widget count that does
    not line up -- was the same `str` in the same `list[str]`, so a caller could not tell "the
    server doesn't know this class, the graph is missing a node" from "this node has one extra
    widget value, which is normal for a seed node's control_after_generate" without parsing
    English. That is what let a mutilated graph reach `/prompt`: the code path had no way to ask
    "was there something serious in there", so it never asked.

    `level`:
      - `"fatal"` -- the conversion is not trustworthy. An unknown node class means the node is
        gone from `prompt` entirely; a widget count off by more than one means more than one
        input landed on the wrong name. Either can produce a graph that executes and returns a
        plausible-looking wrong result -- the exact failure this ticket exists to stop.
        `main()` refuses to submit while any `fatal` Note is outstanding, unless `--force`.
      - `"warn"` -- a real gap, filled automatically (a required input the saved file did not
        specify, taken from the server's own default). Worth seeing, not worth blocking on.
      - `"info"` -- expected/benign: a node the UI itself muted or bypassed, or a widget count
        off by exactly one, which is the ordinary shape of "a node gained one optional widget
        since this file was saved" (see `is_widget`'s CLIPLoader `device` example).
    """

    level: Literal["info", "warn", "fatal"]
    node: str
    text: str


def prompt_to_api(prompt: dict[str, Node]) -> dict[str, dict]:
    return {nid: node.to_api() for nid, node in prompt.items()}


SCALAR_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def is_widget(typ: Any, opts: dict) -> bool:
    """Does this input render as a widget (a value in `widgets_values`)?

    A combo (type given as a list of options) and the primitive scalars are
    widgets. `forceInput: True` demotes a scalar back to a wire-only socket.

    The comma case is not academic. `LTXVEmptyLatentAudio.frame_rate` is typed
    `"FLOAT,INT"` -- a multi-type -- which does not equal any single scalar
    name. Testing `typ in SCALAR_TYPES` dropped it from the widget list, and the
    saved `[49, 25, 1]` then landed as frames_number=49, batch_size=25,
    frame_rate=default: a 25-batch audio render, silently, from a graph that
    would have executed and returned a plausible-looking result. Everything
    after a dropped widget shifts by one, which is the failure this whole
    conversion has to not have.
    """
    if isinstance(opts, dict) and opts.get("forceInput"):
        return False
    if isinstance(typ, list):
        return True
    if isinstance(opts, dict) and opts.get("widgetType"):
        return True
    if isinstance(typ, str):
        return any(part.strip() in SCALAR_TYPES for part in typ.split(","))
    return False


def widget_names(defn: dict) -> list[str]:
    """Ordered names of the inputs that appear as WIDGETS on this node.

    Order comes from `input_order` when the server supplies it -- that is the
    order the frontend laid the widgets out in, and therefore the order the
    saved `widgets_values` are in. Falling back to dict order of required then
    optional is only for nodes that predate `input_order`.
    """
    inp = defn.get("input", {}) or {}
    order = defn.get("input_order") or {}
    names: list[str] = []
    if order:
        for section in ("required", "optional"):
            names.extend(order.get(section) or [])
    else:
        for section in ("required", "optional"):
            names.extend((inp.get(section) or {}).keys())

    out: list[str] = []
    for name in names:
        spec = (inp.get("required") or {}).get(name) or (inp.get("optional") or {}).get(name)
        if not isinstance(spec, (list, tuple)) or not spec:
            continue
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if is_widget(spec[0], opts):
            out.append(name)
    return out


def widgets_com_companheiro(defn: dict) -> set[str]:
    """Widgets que ocupam DUAS posicoes em `widgets_values`, nao uma.

    Um input INT marcado `control_after_generate` e desenhado pelo frontend como dois
    controles -- o valor e o 'fixed/increment/randomize' ao lado -- e o arquivo salvo
    guarda os dois, em sequencia. O servidor so conhece o primeiro.

    Ate 2026-09-12 este arquivo sabia disso em PROSA (o comentario dizia "extras sao
    normais: nodes de seed carregam um control_after_generate") e mesmo assim fazia
    `zip(names, vals)`, que pareia desde o inicio -- entao o extra na posicao 1 deslocava
    TODO o resto. Num KSampler comum isso produzia `steps='randomize'`, `cfg=10`,
    `sampler_name=1.0`, `denoise='simple'`. Aqui deu 400 na validacao e apareceu; com tipos
    compativeis teria RODADO com os valores trocados, que e precisamente a falha silenciosa
    contra a qual o resto deste arquivo foi escrito.
    """
    inp = defn.get("input", {}) or {}
    achados = set()
    for section in ("required", "optional"):
        for nome, spec in (inp.get(section) or {}).items():
            if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                continue
            opts = spec[1] if isinstance(spec[1], dict) else {}
            # `control_after_generate` (seed) e `image_upload` (LoadImage) sao os dois
            # marcadores que o servidor publica para "este widget e desenhado como DOIS
            # controles". Ambos gastam uma posicao extra em `widgets_values`.
            if opts.get("control_after_generate") or opts.get("image_upload"):
                achados.add(nome)
    return achados


# Nodes que existem SO no frontend: nao aparecem em /object_info e nunca deveriam. Sao
# decoracao de canvas, sem entrada, sem saida e sem efeito na execucao. A lista e curta de
# proposito -- ela EXIME de ser fatal, entao um nome a mais aqui esconde um node que sumiu
# de verdade. Qualquer coisa que tenha saida ligada NAO entra nesta lista.
SOMENTE_UI = frozenset({"Note", "MarkdownNote"})


def ui_to_api(wf: dict, object_info: dict) -> tuple[dict[str, Node], list[Note]]:
    """UI-format graph -> {node_id: Node}. Returns (prompt, notes)."""
    notes: list[Note] = []
    # link id -> (origin_node_id, origin_slot)
    links: dict[int, tuple[int, int]] = {}
    for link in wf.get("links", []) or []:
        # [link_id, origin_id, origin_slot, target_id, target_slot, type]
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            links[link[0]] = (link[1], link[2])

    prompt: dict[str, Node] = {}
    for node in wf.get("nodes", []) or []:
        ctype = node.get("type")
        nid = str(node.get("id"))
        if node.get("mode") in (2, 4):  # muted / bypassed
            notes.append(Note(
                level="info", node=nid,
                text=f"node {nid} ({ctype}) is muted/bypassed in the UI -- skipped",
            ))
            continue
        if ctype in SOMENTE_UI:
            # Frontend-only decoration: it has no server class BY DESIGN, carries no input
            # and no output, and its absence from `prompt` changes nothing that executes.
            # Treating it as fatal (the behaviour until 2026-09-12) refused every workflow
            # with a sticky note on the canvas -- including the official Krea2 Edit one,
            # which ships eight of them. That is a fatal that cannot be acted on: there is
            # nothing to fix in the graph.
            notes.append(Note(
                level="info", node=nid,
                text=f"node {nid} ({ctype}) is a UI-only node -- not submitted, by design",
            ))
            continue
        defn = object_info.get(ctype)
        if defn is None:
            # The node vanishes from `prompt` right here -- this is the exact case ticket 03
            # is about. "fatal" is not decoration: `main()` reads this level and refuses to
            # submit, because a graph missing a node can still execute and "succeed".
            notes.append(Note(
                level="fatal", node=nid,
                text=f"node {nid}: server does not know class '{ctype}'",
            ))
            continue

        inputs: dict[str, Any] = {}
        wired: set[str] = set()
        for slot in node.get("inputs", []) or []:
            name, lid = slot.get("name"), slot.get("link")
            if lid is None or lid not in links:
                continue
            origin, oslot = links[lid]
            inputs[name] = (str(origin), oslot)
            wired.add(name)

        # Widgets fill, in order, the widget-able inputs that are NOT wired.
        vals = list(node.get("widgets_values") or [])
        # `widgets_values` e indexado pela lista COMPLETA de widgets do node, incluindo os
        # que viraram entrada ligada por fio -- o frontend guarda o valor velho na posicao.
        # Ler so os nao-ligados desalinha: num EmptySD3LatentImage com width/height ligados,
        # vals=[1024,1024,1] e names=['batch_size'] dava batch_size=1024.
        todos = widget_names(defn)
        names = [n for n in todos if n not in wired]
        # Cada widget com companheiro de UI come uma posicao a mais em `vals`. Contar isso
        # ANTES das comparacoes abaixo, senao um KSampler correto (7 valores, 6 inputs) e
        # lido como "1 extra" e um KSampler de verdade desalinhado passa pelo mesmo buraco.
        companheiros = widgets_com_companheiro(defn)
        esperado = len(todos) + sum(1 for n in todos if n in companheiros)
        if len(vals) < esperado:
            missing = esperado - len(vals)
            # Off by exactly one is the ordinary shape of "a node gained one optional widget
            # since this file was saved" -- the CLIPLoader `device` case this file's own history
            # hit. Off by MORE than one is the failure this ticket is about: more than one input
            # is about to silently land on the wrong name (or at a default it never asked for),
            # which is indistinguishable on screen from a correct run -- worded differently from
            # the benign case below so a printed FATAL line does not read like an INFO one.
            if missing == 1:
                notes.append(Note(
                    level="info", node=nid,
                    text=f"node {nid} ({ctype}): {len(vals)} widget values for {len(names)} "
                         f"widget inputs {names} -- trailing ones left at server default",
                ))
            else:
                notes.append(Note(
                    level="fatal", node=nid,
                    text=f"node {nid} ({ctype}): {len(vals)} widget values for {len(names)} "
                         f"widget inputs {names} -- {missing} widgets missing, check alignment",
                ))
        elif len(vals) > esperado:
            # `esperado` ja contabiliza os companheiros de UI, entao daqui para cima
            # sobra e desalinhamento, nao formato normal.
            notes.append(Note(
                level="fatal", node=nid,
                text=f"node {nid} ({ctype}): {len(vals)} widget values for {esperado} "
                     f"esperados (inputs {names}, companheiros de UI "
                     f"{sorted(companheiros & set(names))}) -- "
                     f"{len(vals) - esperado} sobrando, confira o alinhamento",
            ))
        # Consumir na ordem, pulando a posicao do companheiro LOGO DEPOIS do seu widget --
        # e ali que o frontend a grava, nao no fim da lista.
        i = 0
        for name in todos:
            if i >= len(vals):
                break
            if name not in wired:          # ligado por fio: o valor salvo e lixo, so avanca
                inputs[name] = vals[i]
            i += 1
            if name in companheiros:
                i += 1

        # Fill required widget inputs the saved file does not carry.
        #
        # A workflow saved by an older frontend has fewer widget values than the
        # node now declares -- here CLIPLoader gained a `device` after this file
        # was written, so the graph arrived with 2 values for 3 inputs. ComfyUI
        # does NOT substitute a default for a missing *required* input: it
        # rejects the prompt with 400 "Required input is missing". Taking the
        # default out of /object_info keeps the saved file authoritative for
        # everything it does specify.
        req = defn.get("input", {}).get("required") or {}
        for name, spec in req.items():
            if name in inputs or not isinstance(spec, (list, tuple)) or not spec:
                continue
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if "default" in opts:
                inputs[name] = opts["default"]
            elif isinstance(typ, list) and typ:
                inputs[name] = typ[0]
            else:
                continue
            notes.append(Note(
                level="warn", node=nid,
                text=f"node {nid} ({ctype}): '{name}' absent from the saved file, "
                     f"filled with the server default {inputs[name]!r}",
            ))

        prompt[nid] = Node(class_type=ctype, inputs=inputs, wired=wired)
    # Um node pulado (mutado/bypassado, ou so de UI) sai do `prompt`, mas quem apontava para
    # ele continua com a referencia. O servidor rejeita com 400 e uma mensagem que so diz o
    # ID que falta -- foi assim que o workflow oficial do Krea2 Edit, que traz a SEGUNDA
    # referencia mutada de proposito, dava `exception_during_inner_validation: '90'` em tres
    # nodes ao mesmo tempo, nenhum deles o node 90.
    #
    # Referencia orfa em entrada OPCIONAL simplesmente cai: era exatamente isso que o mute
    # queria dizer. Em entrada OBRIGATORIA nao se pode inventar nada, entao vira `fatal`.
    for nid, node in prompt.items():
        defn = object_info.get(node.class_type, {})
        req = set(((defn.get("input") or {}).get("required") or {}).keys())
        for nome, val in list(node.inputs.items()):
            # `Wire` e tupla, nao lista: um `isinstance(val, list)` aqui nunca casa e a
            # poda vira no-op silenciosa. O node ja sabe quais entradas sao fio.
            if not (node.is_wire(nome) and str(val[0]) not in prompt):
                continue
            if nome in req:
                notes.append(Note(
                    level="fatal", node=nid,
                    text=f"node {nid} ({node.class_type}): entrada OBRIGATORIA '{nome}' "
                         f"aponta para o node {val[0]}, que nao entrou no prompt "
                         f"(mutado, bypassado ou so de UI). O grafo esta incompleto.",
                ))
            else:
                node.inputs.pop(nome)
                notes.append(Note(
                    level="info", node=nid,
                    text=f"node {nid} ({node.class_type}): entrada opcional '{nome}' "
                         f"apontava para o node {val[0]}, que foi pulado -- removida",
                ))

    return prompt, notes


def report(entry: Entry, wall: float, label: str) -> None:
    """Print the human-readable summary of a finished run: status, timings, output files, the
    ticket-04 cache-hit call-out, and the fixed "NOT covered by this run" block. Pure printing --
    the exit-code decision (status != success -> 1, cache_hit -> 5) stays in `main()`, which is
    the only thing here that gets to end the process."""
    print()
    print(f"=== {label} ===")
    print(f"status      {entry.status}")
    print(f"WALL        {wall:.1f}s   (queue -> /history, this server, this cache state)")
    files = entry.files
    print(f"outputs     {len(files)} file(s)")
    for f in files[:8]:
        print(f"            {f}")
    if len(files) > 8:
        print(f"            ... and {len(files) - 8} more")

    server_side_s = entry.server_side_s
    if server_side_s is not None:
        print(f"server-side  {server_side_s:.1f}s "
              f"(execution_start -> execution_success, excludes queue wait)")

    if entry.cache_hit:
        print()
        print("  *** THIS TIMED NOTHING ***")
        print(f"  server-side execution was {server_side_s:.2f}s, under the "
              f"{CACHE_HIT_THRESHOLD_S:.0f}s threshold. That is its per-node result")
        print("  cache answering an identical prompt, not a render. Re-run with")
        print("  --seed <different> to force execution with the weights resident.")
    elif server_side_s is None and files:
        print()
        print("  WARN server sent no execution_start/execution_success messages -- cache-hit")
        print("  detection (ticket 04) could not run; this exit code does not confirm a render.")

    print()
    print("NOT covered by this run:")
    print("  - fairness vs any other engine: storage, cache and load state are NOT controlled here")
    print("  - output correctness: files were written; nobody looked at them")
    print("  - cold vs warm: the first run of a server includes weight load, this does not split it")
    if server_side_s is None:
        print("  - cache-hit detection: server reported no execution_start/execution_success "
              "timestamps for this prompt_id, so a cache hit could slip through as exit 0")


def build_argparser() -> argparse.ArgumentParser:
    """Every flag this tool takes, in one place.

    Extracted because it was two thirds of main()'s length and none of its job. The three
    seams the ticket named -- Comfy, run_and_wait, report -- carve out the WORK; this carves
    out the DECLARATION, which is why main() still did not fit after them.
    """
    ap = argparse.ArgumentParser()
    fonte = ap.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--workflow", help="UI-format workflow (converted via /object_info)")
    fonte.add_argument("--api-prompt", help="API-format graph JSON, submitted as is")
    fonte.add_argument("--lista", help="text file, one API-format graph path per line")
    ap.add_argument("--saida", help="append one JSONL record per graph (--api-prompt/--lista)")
    ap.add_argument("--server", default="127.0.0.1:8190")
    ap.add_argument("--wait-server", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=5400.0)
    ap.add_argument("--dump-api", help="write the converted API prompt here and exit")
    ap.add_argument(
        "--object-info-file",
        help="read /object_info from this JSON file instead of the live server, and skip "
        "waiting for the server. Only meaningful together with --dump-api: the conversion "
        "still needs SOME /object_info, but a real run needs the server up anyway to POST the "
        "prompt, so this flag exists for testing the conversion offline, not for a real run.",
    )
    ap.add_argument("--label", default="run")
    ap.add_argument(
        "--force",
        action="store_true",
        help="submit even if ui_to_api reported a 'fatal' Note (unknown node class, or a widget "
        "count off by more than one). Without this flag, a fatal Note refuses the submission -- "
        "ticket 03: a graph missing a node, or with widgets landed on the wrong names, can still "
        "execute and 'succeed' with a plausible-looking wrong result.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        help="override every seed widget. REQUIRED to time a second run of the "
        "same graph: ComfyUI caches per-node results, so re-posting an identical "
        "prompt returns the previous outputs in well under a second without "
        "executing anything. Changing the seed forces the sampler to re-run "
        "with the weights already resident, which is the actual warm number.",
    )
    return ap


def resolve_object_info(comfy: "Comfy", args: argparse.Namespace) -> dict:
    """The server's node schema, or a file standing in for it.

    Raises TimeoutError if the server never comes up; main() turns that into an exit code. A
    library function does not kill the process -- that was the SystemExit this ticket removed
    from wait_for_server.
    """
    if args.object_info_file:
        # `utf-8-sig` pelo mesmo motivo do `--workflow` em `main()`, e por consistencia: este
        # arquivo tambem chega pela mao do usuario. Aqui nao houve BOM medido -- so o do workflow
        # foi -- entao isto e prevencao, e esta dito como tal.
        object_info = json.loads(Path(args.object_info_file).read_text(encoding="utf-8-sig"))
        print(f"/object_info: {len(object_info)} node classes (loaded from "
              f"{args.object_info_file}, NOT the server -- only valid for offline --dump-api "
              f"testing)")
        return object_info
    waited = comfy.wait_up(args.wait_server)
    print(f"server {args.server} up after {waited:.1f}s")
    object_info = comfy.object_info(timeout=120.0)
    print(f"/object_info: {len(object_info)} node classes")
    return object_info


def apply_seed(prompt: dict[str, "Node"], seed: int) -> int:
    """Force every seed widget to `seed`, and report how many were hit.

    Goes through Node.set_widget, which is a no-op on a wired input by construction -- that is
    the whole reason the model exists (ticket 05). Zero hits is the dangerous case: the graph
    then hits ComfyUI's node cache and times nothing, which reads as a 0.6 s render.
    """
    hit = 0
    for nid, node in prompt.items():
        for key in ("noise_seed", "seed"):
            if node.set_widget(key, seed):
                hit += 1
                print(f"  seed override: node {nid} ({node.class_type}).{key} = {seed}")
    if hit == 0:
        print("  WARN --seed given but no seed widget found -- the graph will "
              "hit ComfyUI's node cache and time nothing")
    return hit


def refuse_on_fatal(notes: list["Note"], force: bool) -> bool:
    """True when a fatal Note must stop the submission. Prints why; the caller picks the code."""
    fatal = [n for n in notes if n.level == "fatal"]
    if not fatal or force:
        return False
    print()
    print(f"REFUSED to submit: {len(fatal)} fatal note(s) from ui_to_api. Submitting this "
          "graph would run a mutilated prompt that can still execute and 'succeed' with a "
          "plausible-looking wrong result (ticket 03). Re-run with --force to submit anyway.")
    for n in fatal:
        print(f"  FATAL node {n.node}: {n.text}")
    return True


def main_api(args: argparse.Namespace) -> int:
    """`--api-prompt` / `--lista`: no conversion, straight to `comfy_client.roda_lista`."""
    comfy = Comfy(args.server)
    try:
        waited = comfy.wait_up(args.wait_server)
    except TimeoutError as e:
        print(str(e))
        return 1
    print(f"server {comfy.base} up after {waited:.1f}s")
    grafos = [Path(args.api_prompt)] if args.api_prompt else le_lista(Path(args.lista))
    return roda_lista(comfy, grafos, Path(args.saida) if args.saida else None, args.timeout)


def main() -> int:
    args = build_argparser().parse_args()
    if args.api_prompt or args.lista:
        if args.seed is not None or args.dump_api or args.object_info_file:
            print("--seed/--dump-api/--object-info-file only apply to --workflow")
            return 2
        return main_api(args)
    # `utf-8-sig`, nao `utf-8`: le os dois casos. Sem BOM os dois codecs sao identicos; COM BOM o
    # `utf-8` entrega ﻿ como primeiro caractere e o `json.loads` levanta
    # `Unexpected UTF-8 BOM (decode using utf-8-sig)` antes de ver um unico no.
    #
    # MEDIDO em 2026-09-01, ao verificar o ticket 07 do ComfyLite: dos 79 workflows do dono em
    # `ComfyUI/user/default/workflows/`, UM tem BOM --
    # `SeedVR2/JOAO_SeedVR2_VIDEO_3080Ti_safe_720p_Q8.json`, primeiros bytes `ef bb bf`. Este
    # conversor nao conseguia abri-lo. Nao e caso hipotetico nem arquivo de teste: e um workflow
    # real, e editores do Windows gravam BOM sem perguntar.
    wf = json.loads(Path(args.workflow).read_text(encoding="utf-8-sig"))
    comfy = Comfy(args.server)

    try:
        object_info = resolve_object_info(comfy, args)
    except TimeoutError as e:
        print(str(e))
        return 1

    prompt, notes = ui_to_api(wf, object_info)
    if args.seed is not None:
        apply_seed(prompt, args.seed)
    for note in notes:
        print(f"  {note.level.upper():5} {note.text}")
    print(f"converted {len(prompt)} nodes")

    if args.dump_api:
        Path(args.dump_api).write_text(json.dumps(prompt_to_api(prompt), indent=2), encoding="utf-8")
        print(f"wrote {args.dump_api} -- not executed")
        return 0
    if refuse_on_fatal(notes, args.force):
        return 4

    try:
        entry = run_and_wait(comfy, prompt_to_api(prompt), args.timeout)
    except ValueError as e:      # Comfy.submit: no prompt_id, or node_errors on an HTTP 200
        print(str(e))
        return 2
    except PollTimeout as e:     # run_and_wait: pid never reached /history in time
        print(str(e))
        return 3

    report(entry, entry.wall, args.label)
    if args.saida:
        registro = {"grafo": args.workflow, "prompt_id": entry.pid, "status": entry.status,
                    "erro": entry.erro, "server_side_s": entry.server_side_s,
                    "wall": round(entry.wall, 3), "cache_hit": entry.cache_hit,
                    "files": entry.files, "onde": entry.onde, "rotulo": args.label}
        registro.update(controles(comfy))
        grava_jsonl(Path(args.saida), registro)
    if entry.status != "success":
        return 1
    if entry.cache_hit:
        return 5  # ticket 04: cache-hit is its own exit code, not folded into 0/success
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
