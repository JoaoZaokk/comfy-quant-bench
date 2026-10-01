"""Atualiza o README do GitHub (o pacote curado do remoto, fundido no master) com o que foi feito
desde 2026-09-01: novos repos no Hub, o audio do LTX, o LoRA sobre 4 bits, e o que esta em curso."""
import io

p = 'README.md'
s = io.open(p, encoding='utf-8').read()

anchor = "## Published checkpoints\n"
assert s.count(anchor) == 1

novo = """## 2026-09-13: five more families, the soundtrack, and what a LoRA becomes in 4 bits

*This repo is now the full bench, not a curated subset.* Everything below is measured on one RTX 3090
(with an RTX 3080 Ti as donor/offload), and every claim carries its evidence on the Hub repo it
belongs to. The running log is [`W4A4_PROGRESS.md`](W4A4_PROGRESS.md); the rules this bench works by,
with every correction it had to make to itself, are in [`CLAUDE.md`](CLAUDE.md); the criteria written
*before* each measurement are in [`bench/criterio_*.md`](bench/).

**Four-bit weights survive everywhere measured; four-bit activations fail in three families of 2026
models, three different ways.** Same 4-bit weights, only the activation path differs:

| model | W4A4 result | W4A8 result | published |
| --- | --- | --- | --- |
| Qwen-Image-Edit 2511 (20.4 B) | pure static | good, 38.05 → 10.79 GiB | [Qwen-Image-Edit-2511-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot) |
| Qwen-Image 2512 | coloured speckle | good, 38.05 → 10.79 GiB, 3.65x faster/step | [Qwen-Image-2512-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Qwen-Image-2512-W4A8-ConvRot) |
| Wan 2.2 TI2V 5B | blur | good, 9.31 → 2.75 GiB, 1.59x faster/step | [Wan2.2-TI2V-5B-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot) |
| LTX 2.5 22B distilled | — (riftcast's W4A4 works) | good, 39.13 → 11.66 GiB, 1.95x faster than BF16 on a 10 s render | [LTX-2.5-22B-distilled-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot) |
| Krea 2 Turbo (12.8 B) | works, ceiling not reached at cg 16 | — | held back by the model's licence gate |

Where the model's author publishes their own INT8, it is the more faithful arm every time (five
families in a row: 1.9x on LTX 2.5's picture, 2.9x on its sound) and ours is ~1.7–1.8x smaller. The
choice is a trade, and each card shows what the trade costs.

**The LTX card measured half the model, and that was corrected.** LTX 2.x generates video *and
audio* in one latent; the first LTX 2.5 card decoded only the frames. Re-rendered with the audio
branch decoded (`tools/ltx_video.py`, measured by `tools/compara_av.py` with silence and white noise
as controls): the soundtrack ranks the arms the way the picture does and by a wider margin —
Lightricks' INT8 at log-mel L1 0.041 against our W4A8's 0.120 (white noise scores 1.47). The MP4s
with their audio tracks and the lossless FLACs are on the Hub repo as proof. The re-render came back
pixel-identical to the first run in all three arms.

**A LoRA loaded onto a quantized weight is a requantization**, not a branch: ComfyUI dequantizes,
adds the delta and requantizes back to 4 bits (`comfy/ops.py:1449-1457`). Measured on the real path
(`tools/probe_lora_requant.py`) across Z-Image, Krea 2, Wan 2.2, LTX 2.5 and Qwen-Image-Edit: the
delta survives in expectation (survival 0.86–1.00), but what lands is the delta plus noise 2–100x its
size, and a no-op requantization already costs 4–14 % extra weight error. A LoRA from the wrong
architecture, or one that ships all-zero `lora_B` tensors (the `ltx2-squish` LoRA does, for every
audio family), still gets its layers requantized — the model gets worse with nothing applied, and the
only trace is a log line. **And the output contradicts the weight:** on Qwen-Image-Edit W4A8 the
Lightning 4-step LoRA — 86 % survival, noise 103x the delta — works completely at the output, with the
control that has to fail (4 steps without it) failing identically on INT8 and W4A8. Weight-space
numbers rank and alarm; the render decides. Criterion and results: [`bench/criterio_lora.md`](bench/criterio_lora.md).

**In progress at the time of this commit:** LTX 2.3 distilled 1.1 (BF16 source on disk, W4A8 and W4A4
converted, the factory Gemma 3 12B encoder converted to W4A8), rendered on the same 10-second
protocol with audio against the BF16 original and a third-party GGUF Q6_K; and the LoRA tests on both
LTX versions. Criterion written first: [`bench/criterio_ltx23.md`](bench/criterio_ltx23.md).

**What this repo does not carry:** the evidence images and videos of each card live on the Hub repo
they belong to (the `bench/hf/*/README.md` files here reference them by relative path), and models are
never committed. The three text-encoder repos on the Hub carry a measured caveat: stock ComfyUI routes
every text encoder's math through the dequantized path, so their 4-bit builds save memory and not time
unless two locks are released.

"""
s = s.replace(anchor, novo + anchor)

# extend the table of published checkpoints with the September repos
old_row = "| [HunyuanVideo-1.5-720p-T2V-Quantized](https://huggingface.co/JoaoZaokk/HunyuanVideo-1.5-720p-T2V-Quantized)"
i = s.index(old_row)
j = s.index("\n", i) + 1
extra = ("| [Qwen-Image-Edit-2511-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot) | *2026-09-13* — the editor measured as an editor: 12 real edits, the three instructions verbatim, W4A4 static published as a negative, the Lightning LoRA test with its control | Apache 2.0 |\n"
         "| [Qwen-Image-2512-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Qwen-Image-2512-W4A8-ConvRot) | *2026-09-13* — 38.05 → 10.79 GiB, W4A4 coloured speckle as a negative | Apache 2.0 |\n"
         "| [Wan2.2-TI2V-5B-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot) | *2026-09-13* — 9.31 → 2.75 GiB, W4A4 blur as a negative, and the counterexample that shows latent divergence is biased toward soft failure | Apache 2.0 |\n"
         "| [LTX-2.5-22B-distilled-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot) | *2026-09-13* — a full 10 s video **with its audio** against BF16 and Lightricks' INT8; MP4/FLAC proofs in the repo | LTX-2.x Community Licence — read it there |\n"
         "| [Qwen2.5-VL-7B-W4A4-ConvRot](https://huggingface.co/JoaoZaokk/Qwen2.5-VL-7B-W4A4-ConvRot) · [Gemma-3-12B-it-Heretic-W4A8](https://huggingface.co/JoaoZaokk/Gemma-3-12B-it-Heretic-W4A8) | text encoders: memory saved, time not, unless ComfyUI's two text-encoder locks are released | Apache 2.0 / Gemma |\n")
s = s[:j] + extra + s[j:]
io.open(p, 'w', encoding='utf-8').write(s)
print('README.md: 2026-09-13 section + rows added')
