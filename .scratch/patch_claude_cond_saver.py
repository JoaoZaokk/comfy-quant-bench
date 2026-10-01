"""CLAUDE.md: duas armadilhas medidas em 2026-09-14 -- o LTXVSaveConditioning perde a chave que o LTX
2.3 exige (render vira ruido), e o s/passo depende do que mais esta residente (3x)."""
import io

p = 'CLAUDE.md'
s = io.open(p, encoding='utf-8').read()
anchor = """is the owner's, not ours: `?:\\pagefile.sys`, system-managed, 61.2 GiB on C: at the time of writing,
and every death above needed it to grow.
"""
assert s.count(anchor) == 1, s.count(anchor)
novo = anchor + """
**And `LTXVSaveConditioning` is not a way to keep the text encoder out of the process for LTX 2.3 —
measured 2026-09-14.** ComfyUI-LTXVideo's saver keeps the tensor and an attention mask. The LTX 2.3
encoder returns `{"unprocessed_ltxav_embeds": True}` beside the tensor
(`comfy/text_encoders/lt.py:201-204`), and the model applies `caption_projection` and the embeddings
connectors only when that key arrives (`comfy/model_base.py:1185` →
`comfy/ldm/lightricks/av_model.py:583`). Loaded back through `LTXVLoadConditioning` the key is gone,
the 6144-wide context passes the "already processed" width check and goes raw into the
cross-attention: the same W4A8 model, same seed, renders brown noise with noise for sound — **MAE
75.9, SSIM 0.19, log-mel 1.05** against the live encoder (`bench/ltx23/cond_identity_ltxv_saver/`).
Four LoRA renders and one BF16 attempt were made on that conditioning before the identity control
caught it; they stay under `bench/ltx23/*_ltxv_saver/` as what the wrong tool produces and decide
nothing. **The identity control is not optional**: a render on saved conditioning that was never
compared with the live path is a render of an unknown prompt. The replacement keeps every option —
`tools/ltx_encode_lowcommit.py` (bare process, zero-commit reader, float32 tensor, options as
tensors and JSON metadata) and `VoidLoadConditioningFull` in `custom_nodes/comfy-void-stage-tools`,
which refuses a file without them; `tools/ltx_video.py --cond-from` uses that node. The encoder run
on the 3080 Ti differs from the server's 3090 encode by rel-L2 1.0e-3 (87 % of elements bit-equal
in bf16, max |Δ| 1.0 on a [−148, 294] range): not bit-identical, and said so where it is used.

**Per-step time depends on what else is resident, by 3x.** The same W4A8 render (249 frames, 8
steps, same seed) measured **79.7 s/step** with the 22.7 GB text encoder loaded live in the same
process and **26.4 s/step** with the encoder out of it (saved conditioning). The mechanism — VRAM
left for activations, the offloaded encoder's RAM, or both — was not isolated. A speed column is
only comparable between arms that share what is resident: every arm of the 2.3 card uses saved
conditioning and no encoder; every arm of the 2.5 card used the live encoder.
"""
s = s.replace(anchor, novo)
io.open(p, 'w', encoding='utf-8').write(s)
print('CLAUDE.md: conditioning-saver trap + residency/speed note added')
