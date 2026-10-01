# Aceitação e rastreadores

> Referência preservada do CLAUDE.md original, linhas 1323–1339, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## Testing a converted model

Structural verification is not acceptance. Every output must additionally pass, in order: normal ComfyUI loader compatibility (real node, not a hand-rolled load), a prompt-encoding smoke test through a long-lived ComfyUI process, and a matched-parameter benchmark against the BF16 source (identical prompt, seed, steps, resolution, sampler, scheduler, frames) recording disk, VRAM, load time, s/it, GPU utilization, power, warnings, and visual quality. Workflows for this live in `ComfyUI/user/default/workflows/` (e.g. `Video-LTX2_MultiGPU.app.json` for the Gemma text encoder).

Record negative and unsupported results rather than hiding them.

## Project tracking

- [W4A4_PROGRESS.md](../../W4A4_PROGRESS.md) — the running log: candidate ranking, completed steps, skipped models with reasons, failed/blocked with exact error strings. **Update it as part of the work**, not afterwards.
- [W4A4_HANDOFF.md](../../W4A4_HANDOFF.md) — current state snapshot and the ordered next-steps list.
- `quantization_inventory.{json,md}` — generated; do not hand-edit.

A second, separate effort also runs on this stand:

- [CORTIQ_LTX25_HANDOFF.md](../../CORTIQ_LTX25_HANDOFF.md) — the `cortiq` / LTX-2.5 investigation: what was **measured** (the 52× end-to-end gap, where the time goes, the PV-NT accuracy result) and the five claims that had to be withdrawn. Read this before touching `F:\cortiq-cmf`.
- [.scratch/estado-entregavel/map.md](../../.scratch/estado-entregavel/map.md) — the **plan**: 17 tickets across both repos, with the blocking graph. Every debt ticket carries its closing criterion, written before anyone looked at the result. `.scratch/` is the local issue tracker; nothing in it touches GitHub.

