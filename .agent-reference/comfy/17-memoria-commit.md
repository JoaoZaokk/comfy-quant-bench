# RAM, commit e processos de download

> Referência preservada do CLAUDE.md original, linhas 1447–1463, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Memory gotchas

Conversions check free disk (`estimate + 1 GiB`) and free RAM (`3 × largest selected tensor + 2 GiB`) and exit rather than thrash. If it refuses, the fix is to close memory-heavy WSL/worker processes **manually** — never change the pagefile or kill processes automatically.

**But that guard measures the wrong quantity for `quant_w4a8.py`, and the real limit is commit, not disk.** Measured 2026-09-21 converting the 46.14 GB `10Eros_v1.5_bf16` (1440 layers): `quant_w4a8.py:286` is a **two-pass** design — *"pass one fills `quantized` with every layer, pass two writes"* — so the whole quantized payload is held in RAM before a single byte lands. Consequences, all observed on this run:

- **No `.partial` exists on disk for most of the conversion.** The `[N/1440] quantized` counter is pass one. Looking for the output file to check progress reads as "nothing is happening" for ~35 minutes.
- The process reached **15.38 GiB private** with **3.45 GiB of free commit** on this host, and the pagefile had grown to 72.06 GiB (limit 135.70, which is *not* the 98.72 recorded elsewhere in this file — it is system-managed and moves, so re-read it every time). It fit, but with no margin.
- The per-layer growth is what makes it predictable: 1056 → 1248 layers cost only 0.57 GiB, so extrapolating from a third of the way through is reliable.

So for a model meaningfully larger than this one, **the commit limit bites before the disk does**, and the existing guard will not catch it because it checks `3 × largest tensor`, not the sum of all of them. Read free commit before starting, not free disk.

**And one harness fact that cost 40 minutes twice the same night:** `nohup … &` launched from the Bash tool **does not survive a turn interruption** — both a 43 GiB and a 24 GiB download died silently, one at 69.3%, with a zero-byte log and no live process. Use the harness's `run_in_background`, which is tracked and notifies. `hf_hub_download`'s `.incomplete` does survive, so a resume costs only the remainder; the way to tell a dead download from a slow one is the `.incomplete`'s **mtime**, not its size.

Known benign noise: `ModelPatcher.__del__` prints an `ON_DETACH` AttributeError on short-lived interpreter shutdown; loading itself still exits 0.


