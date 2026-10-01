# Domain docs

How the engineering skills should consume this repo's domain documentation.

**Layout: single-context.** One `CONTEXT.md` at the repo root and one `docs/adr/` beside it. No
`CONTEXT-MAP.md` — checked 2026-08-22 and found no monorepo signals (no `pnpm-workspace.yaml`, no
`workspaces` field, no populated `packages/*`). The root owns quantization/benchmark tools and
its local custom-node sources; list the current tracked tree instead of quoting the old count.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root.
- **`docs/adr/`** — ADRs that touch the area you are about to work in.

If either does not exist yet, **proceed silently**. Do not flag the absence and do not suggest
creating them upfront; `/domain-modeling` creates them lazily when a term or a decision actually
gets resolved.

## Read these too — they are load-bearing here in a way a generic repo's docs are not

`AGENTS.md` is the single instruction source and `CLAUDE.md` imports it. They carry hard rules that
override defaults (only `python_embeded\python.exe`; never overwrite an original model; never
mass-upgrade the stack; never touch WSL). The historical section *"Say which one it was: traced,
or executed"* is preserved under `.agent-reference/comfy/`; its distinction between reading and
execution still governs how conclusions are recorded.

Then, per area:

| Area | Read first |
|---|---|
| Quantization pipeline, converters, verifiers | `W4A4_PROGRESS.md`, `W4A4_HANDOFF.md` |
| Anything under `F:\cortiq-cmf` or the LTX 2.5 work | `CORTIQ_LTX25_HANDOFF.md` |
| First-block cache | `FBCACHE_FINDINGS.md` |
| Anything an earlier audit already touched | `AUDITORIA_2026-08-18.md` |
| The plan and its tickets | `.scratch/estado-entregavel/map.md` |

## Use the glossary's vocabulary

When output names a domain concept — a ticket title, a refactor proposal, a hypothesis, a test
name — use the term as `CONTEXT.md` defines it. If the concept is not in the glossary, that is a
signal: either the language is being invented (reconsider) or there is a real gap (note it for
`/domain-modeling`).

## The provenance rule applies to domain docs too

A term whose definition came from *reading the code* and a term whose definition came from
*running it* are not the same kind of fact, and this repo has paid for confusing them repeatedly —
five documented cases in `CLAUDE.md`, where a mechanism argument was written as a measurement and
the measurement later disagreed.

So: when a glossary entry or an ADR rests on behaviour nobody executed, **say so in the entry**,
not in the chat. `/domain-modeling` says to cross-reference terms against the code; that
cross-reference is a *trace*, and the entry it produces must be labelled as one.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it rather than silently overriding:

> _Contradicts ADR-0007, but worth reopening because…_

## A note on the sibling project

`ComfyLite/` (a separate git repo living beside `ComfyUI/`) is **not** part of this context. When
it acquires domain language it gets its own `CONTEXT.md` inside its own repo. Do not add ComfyLite
terms to this glossary and do not treat the two as one multi-context repo — they are two repos that
happen to share a parent directory.
