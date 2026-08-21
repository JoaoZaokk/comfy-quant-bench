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
    Run `--warm` twice and compare, or read the two numbers it prints.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


def http_get(base: str, path: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(f"http://{base}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_post(base: str, path: str, payload: dict, timeout: float = 60.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://{base}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # ComfyUI answers a rejected graph with 400 and a JSON body naming the
        # node and the input it did not like. urllib raises before that body is
        # read, so without this the only thing on screen is "HTTP Error 400:
        # Bad Request" -- which says a graph was refused but not what for, and
        # each retry costs another 77 s of server start.
        body = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(body)
            print(f"  server refused ({e.code}): {err.get('error')}")
            for nid, ne in (err.get("node_errors") or {}).items():
                print(f"    node {nid} ({ne.get('class_type')}):")
                for d in ne.get("errors") or []:
                    print(f"      {d.get('type')}: {d.get('message')} -- {d.get('details')}")
        except json.JSONDecodeError:
            print(f"  server refused ({e.code}): {body[:4000]}")
        raise


def wait_for_server(base: str, limit_s: float) -> float:
    """Block until /system_stats answers. Returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            http_get(base, "/system_stats", timeout=5.0)
            return time.time() - t0
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2.0)
    raise SystemExit(f"server {base} did not answer /system_stats in {limit_s:.0f}s")


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
        names = [n for n in widget_names(defn) if n not in wired]
        if len(vals) < len(names):
            missing = len(names) - len(vals)
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
        elif len(vals) > len(names):
            # Extra values are normal: seed nodes carry a UI-only
            # 'control_after_generate' that is not an input. Extras beyond one
            # are worth saying out loud, because a MIS-ALIGNMENT looks the same.
            if len(vals) - len(names) > 1:
                notes.append(Note(
                    level="fatal", node=nid,
                    text=f"node {nid} ({ctype}): {len(vals)} widget values for {len(names)} "
                         f"inputs {names} -- {len(vals) - len(names)} extra, check alignment",
                ))
        for name, val in zip(names, vals):
            inputs[name] = val

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
    return prompt, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
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
    args = ap.parse_args()

    wf = json.loads(Path(args.workflow).read_text(encoding="utf-8"))

    if args.object_info_file:
        object_info = json.loads(Path(args.object_info_file).read_text(encoding="utf-8"))
        print(f"/object_info: {len(object_info)} node classes (loaded from "
              f"{args.object_info_file}, NOT the server -- only valid for offline --dump-api "
              f"testing)")
    else:
        waited = wait_for_server(args.server, args.wait_server)
        print(f"server {args.server} up after {waited:.1f}s")
        object_info = http_get(args.server, "/object_info", timeout=120.0)
        print(f"/object_info: {len(object_info)} node classes")

    prompt, notes = ui_to_api(wf, object_info)
    if args.seed is not None:
        hit = 0
        for nid, node in prompt.items():
            for key in ("noise_seed", "seed"):
                if node.set_widget(key, args.seed):
                    hit += 1
                    print(f"  seed override: node {nid} ({node.class_type}).{key} = {args.seed}")
        if hit == 0:
            print("  WARN --seed given but no seed widget found -- the graph will "
                  "hit ComfyUI's node cache and time nothing")
    for note in notes:
        print(f"  {note.level.upper():5} {note.text}")
    print(f"converted {len(prompt)} nodes")

    if args.dump_api:
        Path(args.dump_api).write_text(json.dumps(prompt_to_api(prompt), indent=2), encoding="utf-8")
        print(f"wrote {args.dump_api} -- not executed")
        return 0

    fatal = [n for n in notes if n.level == "fatal"]
    if fatal and not args.force:
        print()
        print(f"REFUSED to submit: {len(fatal)} fatal note(s) from ui_to_api. Submitting this "
              "graph would run a mutilated prompt that can still execute and 'succeed' with a "
              "plausible-looking wrong result (ticket 03). Re-run with --force to submit anyway.")
        for n in fatal:
            print(f"  FATAL node {n.node}: {n.text}")
        return 4

    client_id = str(uuid.uuid4())
    t0 = time.time()
    resp = http_post(args.server, "/prompt", {"prompt": prompt_to_api(prompt), "client_id": client_id})
    pid = resp.get("prompt_id")
    if not pid:
        print(f"server refused the prompt: {json.dumps(resp)[:2000]}")
        return 2
    print(f"queued prompt_id={pid}")

    last_note = 0.0
    while True:
        if time.time() - t0 > args.timeout:
            print(f"TIMEOUT after {args.timeout:.0f}s -- prompt {pid} still not in /history")
            return 3
        try:
            hist = http_get(args.server, f"/history/{pid}", timeout=30.0)
        except (urllib.error.URLError, OSError, TimeoutError):
            hist = {}
        if pid in hist:
            break
        now = time.time()
        if now - last_note >= 30.0:
            last_note = now
            try:
                q = http_get(args.server, "/queue", timeout=10.0)
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
                print(f"  [{now - t0:6.1f}s] running={running} pending={pending}")
            except Exception:
                print(f"  [{now - t0:6.1f}s] waiting")
        time.sleep(2.0)

    wall = time.time() - t0
    entry = hist[pid]
    status = (entry.get("status") or {}).get("status_str", "?")
    outputs = entry.get("outputs") or {}
    files = [
        f"{v.get('subfolder','')}/{v.get('filename','')}"
        for out in outputs.values()
        for key in ("images", "gifs", "audio", "video")
        for v in (out.get(key) or [])
    ]

    print()
    print(f"=== {args.label} ===")
    print(f"status      {status}")
    print(f"WALL        {wall:.1f}s   (queue -> /history, this server, this cache state)")
    print(f"outputs     {len(files)} file(s)")
    for f in files[:8]:
        print(f"            {f}")
    if len(files) > 8:
        print(f"            ... and {len(files) - 8} more")

    # Per-node execution time, if the server recorded it.
    msgs = entry.get("status", {}).get("messages") or []
    starts = {}
    for m in msgs:
        if not (isinstance(m, (list, tuple)) and len(m) >= 2):
            continue
        kind, data = m[0], m[1]
        if kind == "execution_start":
            starts["_t0"] = data.get("timestamp")
        elif kind == "execution_success" and "_t0" in starts and data.get("timestamp"):
            print(f"server-side  {(data['timestamp'] - starts['_t0']) / 1000.0:.1f}s "
                  f"(execution_start -> execution_success, excludes queue wait)")

    # A cache hit and a fast render are the same two numbers on screen.
    if wall < 5.0 and files:
        print()
        print("  *** THIS TIMED NOTHING ***")
        print("  ComfyUI returned outputs in under 5 s. That is its per-node result")
        print("  cache answering an identical prompt, not a render. Re-run with")
        print("  --seed <different> to force execution with the weights resident.")

    print()
    print("NOT covered by this run:")
    print("  - fairness vs any other engine: storage, cache and load state are NOT controlled here")
    print("  - output correctness: files were written; nobody looked at them")
    print("  - cold vs warm: the first run of a server includes weight load, this does not split it")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
