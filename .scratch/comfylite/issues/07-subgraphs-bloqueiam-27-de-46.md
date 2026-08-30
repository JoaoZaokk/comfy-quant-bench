# 07 - Subgraph expansion is unimplemented, and it blocks 27 of the owner's 46 workflows

Type: grilling
Status: ready-for-human
Blocked by: -
Provenance: EXECUTED 2026-08-23 against a RUNNING ComfyUI 0.33 on the 3090

## What was measured

ComfyUI was started for the first time in this project's life and every one of the 46 real workflow
files was converted against the **live** `/object_info` (3366 node classes):

```
conversion over 46 real workflow files, live object_info:
  fatal              32
  CONVERTS           14

why the fatals (grouped):
   27x  node N is an instance of subgraph '<uuid>'
    4x  widget count mismatch (ByteDance2TextToVideoNode, PreviewAny, ...)
    1x  other
```

**59% of the owner's collection cannot run.** Every Z-Image, Qwen-edit and Flux template on this bench
is a subgraph instance — those are exactly the light, fast, everyday workflows, so the ones that DO
convert are the heavy video ones.

The converter refuses loudly rather than silently mangling, which is right:

```
node 57 is an instance of subgraph 'f2fdebf6-...'; expanding subgraphs is not implemented here,
so its contents never reach the server
```

## Why this is not a small fix

A subgraph instance is a node whose `type` is a uuid naming an entry in the file's own
`definitions.subgraphs`. Expanding it means inlining that definition's nodes with fresh ids, rewiring
its inputs and outputs to the instance's links, and doing it recursively — a subgraph may contain
another. Widget promotion makes it worse: a subgraph exposes selected inner widgets on its instance,
so `widgets_values` on the instance maps to inner nodes by a table, not by position.

That last part is where the danger is, and it is the same danger this project keeps meeting: a
half-correct expansion produces a graph that RUNS and returns a plausible wrong image.

## Two related bugs, both found by running, both already fixed

**A. `SaveVideo.codec` was dropped, and the cost was 97.7 s of GPU.** ComfyUI 0.33 types that input
`COMFY_DYNAMICCOMBO_V3` with no `widgetType` option, so `is_widget()`'s string branch classified it as
a link. `widget_names` returned `['filename_prefix', 'format']` for three saved values, `codec` never
reached the prompt, and the run died at the last node on
`SaveVideo.execute() missing 1 required positional argument: 'codec'` — **after** SDXL's 15 steps and
SVD's 20. Fixed two ways: `is_widget` now treats any type whose name contains `COMBO` as a widget
(substring, not a list — the suffix already moved V1→V3), and a required WIDGET input with no value
and no default is now a **fatal** note instead of a silent `continue`.

The first version of that fatal was too broad and failed 16 tests by firing on unwired `MODEL` /
`CONDITIONING` inputs in single-node fixtures. **The tests were right and the code was wrong**;
narrowing it to widget-typed inputs is the fix.

**B. `/history` is empty on this install, so a successful run reported zero outputs.** EXECUTED: a
prompt finished `done` in 93.4 s and wrote `ComfyUI/output/video/ComfyUI_00023_.mp4`, while
`/history/{prompt_id}` **and** `/history?max_items=3` both returned `{}`. Output discovery went only
through history, so the UI had nothing to show for a run that worked. Fixed by also collecting from
the `executed` WebSocket frame, which carries the filenames (TRACED,
`ComfyUI/execution.py:436` and `:578`). Both sources are unioned, history first — neither is trusted
alone.

After both fixes, run 4: **`done` in 224.1 s, 2 outputs**, a 1,371,698-byte PNG (magic
`89504e470d0a1a0a`) and a 468,571-byte MP4 (magic `...66747970`), both fetched back through
ComfyLite's own `/api/output` proxy.

## A third finding, NOT fixed

`probe()`'s 4 s default timeout turned a **busy** ComfyUI into `503 timed out after 4s -- a socket
opened but nothing answered`, refusing to queue. Measured moments later, `/system_stats` answered in
1.09 s; the slow reading was transient. ComfyUI has a queue precisely so work can be submitted while
it is busy, so refusing to submit because a status probe was slow is the wrong call. Decide whether
`/api/generate` should submit anyway when the last known state was reachable.

## Closing criterion

Closed when the owner decides between:

- **A. Implement subgraph expansion** — the only path that makes the everyday workflows usable. Needs
  recursion, id remapping, and the widget-promotion table, and it needs a test that proves a converted
  subgraph produces the SAME prompt ComfyUI's own frontend would submit, not merely a prompt that runs.
- **B. Leave it refused** — 14 workflows work, the rest say why. Honest, and useless for the light
  ones.
- **C. Convert through ComfyUI's frontend instead** — ask a running ComfyUI to flatten the graph rather
  than reimplementing its expansion. Not investigated; there may be no such endpoint.

Recommendation: **A**, and treat "same prompt the frontend would submit" as the acceptance test rather
than "it ran".
