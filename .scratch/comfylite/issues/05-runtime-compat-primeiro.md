# 05 - Compatibility Runtime first: what ComfyLite can reuse, and the three recorded failures of the alternative

Type: grilling
Status: resolved
Blocked by: 01
Provenance: TRACED

## Question

Handoff section 18 defines two runtimes. Which one does the MVP build?

## The recommendation is Compatibility, and the bench has evidence

`tools/comfy_run_workflow.py` (685 lines) is the real asset and it already solves the hard part:
converting a saved **UI-format** workflow to **API format** and posting to `/prompt`. Its docstring
states the constraint that would otherwise be rediscovered painfully:

> the conversion must fetch `/object_info` from *the same server that will run the prompt*, because
> widget order is a property of that server's loaded node definitions. Reading a stale
> `object_info.json` off disk is how a graph runs with parameters shifted by one slot and still
> completes.

**"Still completes" is the whole problem.** A ComfyLite that caches `/object_info` produces images that
are wrong rather than errors.

Against that, `tools/ltx25_queue.py:5-14` is a three-item list of **recorded, executed failures** of
driving a model from a standalone script on this bench -- one of which surfaced only *after* 39 GiB had
been read into RAM. Read it before anyone proposes handoff section 13's Optimized/Direct Runtime.

## What is missing from the reusable half

`comfy_run_workflow.py` **polls**. `history()` at `:212-218` and `queue()` at `:224`. Grep for
`websocket` or `/ws` over the file: no hits. Handoff section 3's "progress/events" needs ComfyUI's `/ws`
channel, which nothing on this bench touches. That is the one piece of genuinely new work in phase 3.

`ltx_studio.py` is worth reading as a **precedent that a local page is enough for a viewer** -- it
parses `step N/M` for a real progress bar and states plainly when a phase has no fraction to report --
and worth reading as a **list of what made the owner reject it**: 13 minutes for 2 seconds of video,
output in 49 PPM files nothing opens, half the render silent. Its own footer says
*"NOT covered: this is a shell over one subcommand. No graph, no LoRA, no nodes."*

## What ComfyLite inherits for free from the resolver side

`tools/precheck_workflows.py` checks two of handoff section 9's twelve detections: a node type that does
not exist in this installation (against `/object_info`), and a widget value naming a model that is in no
search path. Missing: dependency graph, custom-node install, Python packages, version and quantization
incompatibility, renamed nodes, and the one the handoff leans on hardest -- **"modelos equivalentes ja
instalados"**. Nothing on this bench proposes a substitute.

## And the piece the handoff is missing entirely

`tools/dispatch_census.py:1-13` counts, **during a real generation**, which dispatch branch every
quantized Linear actually took -- because `comfy_kitchen/tensor/convrot_w4a4.py:237` silently falls back
to `F.linear(input, weight.dequantize(), bias)` under one condition.

Handoff section 19 wants PASS / DEGRADED / FAIL. **A PASS without a dispatch census is not a PASS** --
that is the whole content of the memory `ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho`, where four
times in one day the difference measured was which dispatch each arm went through, not the change under
test. An attention autotuner without a census is that failure waiting to happen, and the handoff does
not mention it.

## Closing criterion

Closed on a decision: Compatibility-first or not, and whether the census requirement goes into the
handoff. No code required to close.

---

## DECIDIDO PELO DONO, 2026-09-01: COMPATIBILITY

Palavras dele: *"mvp e compatibility"*. Fecha do jeito que a recomendacao pedia, e sem ressalva.

Consequencia imediata, ja registrada no ticket 04: o Compatibility Mode **nao toca a placa** -- o
ComfyUI ja segura o cartao como um processo so. Entao a participacao no lock (`04`) deixa de ser
requisito do MVP e vira requisito das fases que tocam GPU (autotuner de atencao, analisador de
LoRA, planejador multi-GPU, ProbeRunner).
