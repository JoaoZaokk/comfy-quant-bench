# Leitura versus execução e evidência

> Referência preservada do CLAUDE.md original, linhas 610–657, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Say which one it was: traced, or executed

Reading code and running code produce the same confident prose. That is the specific failure mode
this project keeps hitting — not carelessness, and not reluctance to be wrong, but that a careful
trace through a call chain *feels* like evidence and *reads* like a measurement, and nothing in
the writing separates them. It has cost real work here more than once:

| written as fact | what it actually was | what measurement said |
|---|---|---|
| "crest factor is the statistic ConvRot's activation path is sensitive to" | a mechanism argument, in a tool's own docstring | Spearman **+0.096** over 170 layers. No relationship. |
| "CUDA graph removes the 109 us" | inference from what graphs do | removes **83%**; ~11 us of host survives per replay |
| "host overhead is 161 us" | one median, from a contaminated process | **77 us** clean. Passed to a sibling project before it was checked. |
| "`verify_w4a4.py` runs the same check" | a claim in this very file | it resolves one op; the converter resolves two |
| "the probe is exact, with no error at all" | true of the weight, argued for the activation | BF16 scale rounding, ~1.4e-3 mean relative |

The fix is mechanical, because intent does not survive the next session. **When a conclusion comes
from reading, the artifact that carries it must say so, in the artifact.** Not in the chat, which
is gone by then.

- Tools print it. `tools/verify_w4a4.py`, `tools/test_svdq_verify.py` and
  `custom_nodes/comfy-quant-preflight/` each end their output with what they did *not* cover, on
  every run, pass or fail — a column of PASS lines otherwise reads as "verified".
- Docstrings carry the provenance and the caveat inline, next to the number, not in a paragraph
  below it. Numbers get pasted out of this repo into other sessions; a ratio with its condition
  attached ("4.89x [4.72-5.06] at M=5856 on weight [3840, 3840]; 1.5-2.0x *slower* at M=1")
  cannot be misquoted the way a bare "4.6x" can.
- **A single run is not a measurement.** Three consecutive `m_crossover` runs on an idle, locked
  3090 disagreed by up to 1.4x at the same M and shape, and the crossover itself moved a step
  between two runs in the *same* direction (2026-08-19, `W4A4_PROGRESS.md` part 11). The tool now
  repeats interleaved and prints the ratio's own min-max, because a two-decimal number from one
  burst claims a precision this bench does not have. Anything quoted from here needs a repeat
  behind it and the spread beside it.
- An unverified finding stays labelled unverified all the way into the file that acts on it, and
  **the label changes when the measurement arrives** — in both directions. The preflight package
  started with two audit-derived WARNs. `check_full_precision_matrix_mult` still says "not
  confirmed by execution" in the message itself, because a check that blocks on a hypothesis
  teaches people to disable checks. `check_lora_over_quantized` no longer says it: the hypothesis
  was **measured on 2026-08-19 and did not reproduce** — 680 of 680 dispatches stayed on the
  native kernel with the LoRA applied and in effect (latent norm moved 747.06 → 728.99, so it was
  genuinely applied). It is still a WARN, narrowed to what is actually still unknown: whether the
  delta costs *accuracy* once the weight is already 4-bit, which nobody asked.
- **No upstream PR from a trace.** Only from something run here. `AUDITORIA_2026-08-18.md` lists
  three PR candidates; only one has been proved, and the other two wait for the GPU.

Corollary that has paid off repeatedly: when a measurement contradicts a claim in this repo, the
claim is what changes, including claims in this file — and the tool that produced the wrong number
gets fixed too, not just the sentence.

