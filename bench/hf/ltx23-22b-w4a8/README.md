---
license: other
license_name: ltx-2-community-license-agreement
license_link: https://huggingface.co/JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot/blob/main/LICENSE_LTX_2_COMMUNITY.txt
base_model: Lightricks/LTX-2.3
base_model_relation: quantized
library_name: diffusion-single-file
tags:
  - comfyui
  - text-to-video
  - video
  - audio
  - quantized
  - int4
  - convrot
  - w4a8
language:
  - en
---

# LTX 2.3 22B distilled 1.1 — 4-bit weights in the single-file checkpoint, measured on 10 seconds of video **and audio**

A `asym_w4a8_int8` build of `ltx-2.3-22b-distilled-1.1`, the single-file checkpoint (transformer +
video VAE + audio VAE + vocoder + text projection), measured on a **complete 10-second video with
its soundtrack** against the BF16 original, against a pure W4A4 build of the same source, and
against a third-party 6-bit GGUF.

**42.98 GiB → 15.51 GiB, 2.77x lighter for the whole checkpoint** (the transformer is what shrinks;
the VAEs, vocoder and projection are preserved byte-for-byte, 1503 tensors, verified by hash).

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**
The 2.5 sibling, measured on the same protocol: [LTX-2.5-22B-distilled-W4A8-ConvRot](https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot).

---

## The measurement: one 10-second video, four transformers, picture and sound

249 frames at 25 fps (**9.96 s** — LTX requires `8n+1`), 512×512, **8 steps** (Lightricks' README:
*"distilled v1.1, 8 steps, CFG=1"*; the sigmas are the first stage of ComfyUI's own
`video_ltx2_3_t2v` template: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0`),
cfg 1.0, euler, seed 1234, the same prompt as the 2.5 measurement, one RTX 3090.

**The transformer is the only thing that changes.** Every arm uses the factory text encoder
(`gemma_3_12B_it`, Comfy-Org's BF16 repack, **not** an abliterated variant), the same video VAE,
audio VAE, vocoder and text projection — all read from files whose tensors are byte-identical to
the BF16 checkpoint's.

### Picture

| arm | GiB on disk | sampler, 8 steps | whole run | MAE vs BF16 | PSNR | SSIM | motion |
|---|---|---|---|---|---|---|---|
| BF16 original (lossless GGUF container, transformer only) | 39.15 | 8.05 s/it (66 s) | 155 s | — | — | — | 1.81 |
| GGUF Q6_K, third party (transformer only) | 16.55 | 5.04 s/it (40 s) | 91 s | **3.59** [3.26–4.25] | **29.00 dB** | **0.941** | 1.86 |
| **this file, W4A8** (single-file checkpoint) | **15.51** | **2.30 s/it (18 s)** | 208 s | 10.39 [9.94–10.99] | 21.89 dB | 0.829 | 2.02 |
| W4A4, same source (control; not published) | 14.31 | 1.59 s/it (12 s) | 219 s | 14.45 [13.26–15.32] | 20.36 dB | 0.734 | 2.45 |

MAE is the mean absolute per-pixel difference on the 0–255 scale over **all 249 frames**, with the
per-frame range in brackets. Motion is the mean |frame_t − frame_t−1|: nobody froze; the W4A4 arm
moves a little more than the rest. **"Sampler" is the eight sampling steps as the server's own
progress bar timed them — the only column that is generation speed.** "Whole run" is everything
from the request to the MP4 and is dominated by where the file was read from, not by the
transformer: the two GGUFs came from local disks and the two safetensors from a network share,
which is why the 39 GiB reference "ran" faster than the 15.5 GiB W4A8. The BF16 arm's sampler time
is that of a model **partially loaded** (20.7 GB on the card, 19.6 GB streamed from RAM every
step); its 8.05 s/it is reported, not compared. The 15.51 GiB of this file includes the VAEs,
vocoder and projection (3.85 GiB); the transformer alone is 11.66 GiB against the 39.15 of BF16 —
**3.4x smaller, 3.5x faster per step than the streamed BF16, 2.2x faster than the 6-bit GGUF.**

### Sound

| arm | log-mel L1 vs BF16 | SNR | spectral convergence | lag | level |
|---|---|---|---|---|---|
| GGUF Q6_K, third party | **0.163** | −1.0 dB | 0.419 | 0.0 ms | −24.1 dBFS |
| **this file, W4A8** | **0.163** | −0.9 dB | 0.424 | 0.0 ms | −24.0 dBFS |
| W4A4, same source | 0.281 | −0.8 dB | 0.491 | 0.0 ms | −25.8 dBFS |
| *control: silence* | 8.460 | 0.0 dB | — | — | — |
| *control: white noise at the reference's RMS* | 1.748 | −3.0 dB | 0.986 | — | — |

Reference −23.5 dBFS, 0 % silence, 9.93 s at 48 kHz. Log-mel L1 is the mean absolute difference of
64-band log-mel spectrograms; SNR is waveform-level and phase-sensitive (all three arms sit near
−1 dB because eight sampling steps decorrelate the phase, and the lag column shows nobody slid in
time); level is RMS. Proofs: `av/*.mp4` (with the track), `av/*.flac`, `av/contato_av.png`,
`av/comparacao_av.json`.

### What it says

- **W4A8 is a usable 2.3 at 3.4x less transformer** — the same lighthouse, rocks, waves and birds
  as the reference, the same soundtrack envelope, level within 0.5 dB, no time shift
  (`av/w4a8_this_file.mp4`). It is **not** the closest arm: the 6-bit GGUF is 2.9x nearer in the
  frames (3.59 against 10.39) with high-precision dequantized math — the direction this bench has
  measured every time more weight bits met a 4-bit build — and pays for it with a 2.2x slower step
  and a transformer 1.4x larger.
- **The sound does not separate W4A8 from Q6_K at 8 steps — and that is the metric saturating,
  not the audio being insensitive.** 0.163 against 0.163 in log-mel, 0.42 against 0.42 in spectral
  convergence, waveform SNR near −1 dB for every arm: after eight steps both quantized arms' audio
  has drifted out of phase with the reference and the log-mel distance lands on the same floor.
  Re-run at 3 steps (the tail of the same schedule, same conditioning), the sound separates
  **3.5x** — Q6_K 0.063 with SNR +8.4 dB (still in phase), W4A8 0.223 with SNR −1.1 dB — and the
  picture keeps the same order (7.5 against 13.3). Caveat printed with the number: at 3 steps this
  model does not render its normal scene (every arm, BF16 included, draws bare branches and birds
  with the lighthouse hidden), so the 3-step run answers only the question about the tie, nothing
  about the model as used (`av_3steps/` in the method repo). Listen to the FLACs.
- **W4A4 does not break** (same scene, coherent motion, `av/w4a4_control.mp4`), unlike every Qwen
  and Wan build on this bench, and it is worse than W4A8 in both branches — 14.45 against 10.39 in
  the frames, 0.281 against 0.163 in the sound, and it drops the level by 2.3 dB. It has the fastest
  step (1.59 s/it) and it is not published; the W4A8 is the trade this card proposes.
- **What is common to all three arms** — the ~2 MAE the conditioning path contributes (measured
  below) — is smaller than every gap in the table.

## The conditioning: encoded once, fed to every arm, and checked against the live encoder

The 22.7 GB Gemma 3 12B encoder cannot sit in the same process as a 22 B transformer on this
machine (see the commit-charge section below), and loading it for every render costs about seven
minutes of wall-clock per run — the same W4A8 render took 638 s with the encoder loaded live and
208 s with the conditioning read from a file, while the sampler's own eight steps took 18 s in
both. So the prompt was encoded **once**, in a separate process on the second GPU
(`tools/ltx_encode_lowcommit.py`: read-only reader, float32 output, every option the encoder
returns stored beside the tensor), and every arm below reads that file through
`VoidLoadConditioningFull` (`custom_nodes/comfy-void-stage-tools` in the method repo).

**Why not the stock saver.** ComfyUI-LTXVideo's `LTXVSaveConditioning` / `LTXVLoadConditioning`
keep the tensor and its attention mask and drop the rest. For LTX 2.3 the rest is the whole point:
the encoder returns `{"unprocessed_ltxav_embeds": True}` beside the tensor, and the model applies
`caption_projection` and the audio/video embedding connectors **only when that flag arrives**.
Loaded back without it, the 6144-wide context passes the "already processed" width check and goes
raw into the cross-attention. Same model, same seed, same prompt: **brown noise with noise for
sound — MAE 75.9, SSIM 0.19, log-mel L1 1.05** against the live encoder
(`conditioning/NEGATIVE_ltxv_saver_drops_the_key.png`). Four LoRA renders and one BF16 attempt were
made on that conditioning before the control caught it; none of them is used here.

**The control that had to pass.** The W4A8 transformer, 249 frames, seed 1234, conditioning from
the file against the same render with the encoder live in the process
(`conditioning/identity_saved_vs_live_encoder.png`, `.json`):

| | video MAE | PSNR | SSIM | log-mel L1 | SNR | lag | RMS |
|---|---|---|---|---|---|---|---|
| saved conditioning vs live encoder | **1.74** [1.46–2.08] | 35.7 dB | 0.977 | **0.074** | 4.5 dB | 0 ms | −24.0 vs −23.9 dBFS |
| control: silence | | | | 8.38 | 0.0 dB | | |
| control: white noise, same RMS | | | | 1.75 | −3.0 dB | | |

Not zero, and it should not be: the file was encoded on a different card (RTX 3080 Ti; the
live arm encoded on the 3090), and the two encodes differ by relative L2 1.0e-3 (87 % of the
elements bit-equal in bf16), which eight sampling steps amplify into a visible but small
difference. **Read the arms below with that footprint in mind: a difference under about 2 MAE
between two arms is inside what the conditioning path alone contributes.** Every arm shares the
same file, so the footprint is common to all of them.

**What this does to the speed columns.** Every arm here runs with no text encoder in the process,
and the table keeps two times apart on purpose: the **sampler's eight steps** as the server's own
progress bar timed them, and the **whole run** — model load from a network share, conditioning
load, sampling, both VAE decodes, muxing. Only the first is generation speed. The 2.5 card's
"s/frame" column was the second kind and is labelled so now.

## How the BF16 reference was rendered, and what killed six attempts at it

The reference arm is the unquantized transformer: 39.13 GiB of BF16. On this machine — 63.6 GiB of
RAM, a 61 GiB system-managed pagefile, about 70 GiB of commit already taken by other processes —
six attempts to load it through ComfyUI's safetensors reader killed the server:

| # | loader | file read from | text encoder in the process | died in |
|---|---|---|---|---|
| 1–2 | `CheckpointLoaderSimple` + DisTorch2 `cpu,40gb`, the 43 GiB single file | W: (SMB share) | yes, live, 24.4 GB | `torch/storage.py __getitem__` inside `load_torch_file` |
| 3–4 | transformer extracted to its own file, `UNETLoaderDisTorch2MultiGPU` | W: | yes | same |
| 5 | same loader, saved conditioning, no encoder | C: (local NVMe) | no | `nn.Linear.__init__` — the model's `torch.empty` |
| 6 | dynamic VRAM on (lazy `Linear`, aimdo reader), no DisTorch | C: | no | `HostBuffer.read_file_slice failed`, `cudaErrorMemoryAllocation` |

Five of the six were `Windows fatal exception: access violation`. The network share was the first
suspect and it was cleared by measurement (2026-09-14, bare Python processes, system commit
counters read around each call): `safetensors.safe_open(framework="pt")` maps the file
**copy-on-write twice** — memmap2 for the header, `torch.UntypedStorage.from_file(shared=False)`
for the data — and Windows charges a copy-on-write view its whole size at mapping time, so opening
this 39 GiB file costs **+80.2 GiB of commit** before a tensor is read (+40.8 once the header view
is dropped), and building the model's parameters costs another +40.7. A read-only mmap costs 0.
When the charge forces the pagefile to grow, the new view sometimes comes back with the limit raised
but the charge not taken, and the first read through it is the access violation above — reproduced
in 20 seconds without ComfyUI, on the local drive as well as on the share; the same call survives
on other runs, which is why one such load in seven ever succeeded. A seventh attempt through the
same reader was not made: it would have needed 80 to 120 GiB of commit against about 53 free.

The way out was to not use that reader. `tools/safetensors_to_gguf_bf16.py` writes the same BF16
tensors into a GGUF container through a read-only memmap, with the type policy of ComfyUI-GGUF's
own converter (1-D, small, `scale_shift_table` and `learnable_registers` tensors in F32, exact
from BF16; everything else BF16) and the third-party Q6_K file as a template — 4444 of 4444 names
and shapes agree, and the F32 tensors are byte-identical to theirs in 32 of 32 sampled.
`UnetLoaderGGUF` reads it through `numpy.memmap` and assigns the tensors without `torch.empty`.
`tools/probe_gguf_bf16_equivalence.py` then checked, on 12 sampled Linear layers on the GPU, that
the bytes equal the safetensors', that ComfyUI-GGUF's BF16 dequantization returns the original
bf16 tensor bit for bit, and that `GGMLOps.Linear` and `comfy.ops.manual_cast.Linear` give
identical outputs for the same input, bias included: **12 of 12, maximum difference 0.0**. So the
reference arm is the BF16 model, executed through the GGUF loader; what that loader changes is
only where the weights live between steps (streamed from the page cache instead of resident),
which is a speed axis — its s/frame is reported and not compared.

## LoRAs on this file, measured at the output twice

A LoRA loaded onto a quantized weight is a **requantization**: ComfyUI dequantizes the layer, adds
the low-rank delta and requantizes back to 4 bits (`comfy/ops.py:1449-1457`). On this file, with
`LTX23_Product_Commercial_LoRA` (rank 16, trigger word `srx_commercial`, 1632 target layers, all
matched, no shape failures), the weight-space probe says the delta survives in expectation
(survival 0.908, range 0.87–0.93 over 19 sampled layers) but lands with noise about 20x its own
size (cosine 0.046 between what was asked and what landed), and a no-op requantization alone
raises the per-layer weight error from 0.0731 to 0.0837. Those are weight-space numbers: they rank
and alarm, they do not decide. The output does.

Two rounds of 49 frames, seed 1234, this W4A8 file, the same saved conditioning in all four arms
of each round, `LoraLoaderModelOnly` (merged into the weight) against `LoraLoaderBypassModelOnly`
(delta kept as a BF16 branch, weight untouched), and the same prompt at another seed as the control
that says what "a different video" costs:

| round | arm | MAE vs no-LoRA, same seed | SSIM | motion | log-mel L1 | lag | level |
|---|---|---|---|---|---|---|---|
| lighthouse prompt, no trigger | LoRA merged | 22.9 | 0.644 | 2.91 | 0.543 | −0.1 ms | −12.0 dBFS |
| | LoRA bypass | 24.0 | 0.632 | 2.96 | 0.613 | −17.7 ms | −14.1 dBFS |
| | no LoRA, seed 4321 | 81.0 | 0.452 | 1.91 | 1.083 | −85.9 ms | −12.2 dBFS |
| | merged vs bypass | **5.1** | 0.894 | | 0.232 | 0.0 ms | |
| `srx_commercial` headphones prompt | LoRA merged | 40.9 | 0.702 | 4.55 | 0.588 | −0.3 ms | −14.9 dBFS |
| | LoRA bypass | 40.6 | 0.705 | 4.70 | 0.719 | +14.8 ms | −17.0 dBFS |
| | no LoRA, seed 4321 | 57.5 | 0.699 | 5.88 | 1.747 | +41.5 ms | −11.8 dBFS |
| | merged vs bypass | **7.5** | 0.870 | | 0.328 | 0.0 ms | |

Reference levels −11.9 and −12.3 dBFS; reference motion 2.0 and 1.25. Proofs: `lora/*.mp4`,
`lora/contato_av.png`, `lora/comparacao_av.json`, `lora/merged_vs_bypass.json`.

- **The LoRA arrives at the output.** With the trigger word both LoRA arms execute the prompt's
  *"rotating slowly"* — side view in frame 1, front view by frame 49, motion 4.6 against the
  reference's 1.25 — and draw a different, more refined headphone; the no-LoRA reference on the
  same seed barely turns. Without the trigger the LoRA keeps the lighthouse composition and changes
  the finish (clearer sky, more birds). Sound follows: 0.54–0.59 log-mel from the reference
  against 1.08–1.75 for a seed change.
- **Merged and bypass land on the same video.** 5.1 and 7.5 MAE apart (SSIM 0.87–0.89) against
  23–41 from the reference: the requantization noise, 20x the delta in the weight, is worth a fifth
  of the LoRA's effect at the output. The two rows of each round are hard to tell apart on the sheet.
- **A seed change still moves more than the LoRA does** (57 against 41 with the trigger, 81
  against 23 without). The prediction written before the round — that with the trigger the LoRA
  would move the output more than a seed does — was refuted, as it had been on 2.5.
- **The bypass loader lowers the audio level both times** (−2.2 and −4.7 dB against the reference;
  merged −0.1 and −2.6 dB) and shifts it (−17.7 and +14.8 ms; merged within 0.3 ms). Two
  observations in one direction, no mechanism isolated.
- **Speed.** Merged costs nothing per step (0.55 s/it, the kernel is unchanged); bypass costs 24 %
  (0.68–0.69 s/it) for the extra low-rank branch. 49 frames, 8 steps, sampler time from the
  server's progress bar.

Not covered: strength 1.0 only; one LoRA; the same LoRA was not rendered on the BF16 original, so
whether its effect matches the unquantized model's is not measured; text-encoder LoRAs not tested.

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.3-22b-distilled-1.1_w4a8.safetensors` | 16,651,422,654 | **15.51** | 1440 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256; 4507 tensors preserved |

Source: `ltx-2.3-22b-distilled-1.1.safetensors`, 46,149,345,334 B, sha256
`b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc`, byte-for-byte the file
`Lightricks/LTX-2.3` publishes. Backend recorded in the sidecar: `comfy_kitchen.backends.cuda`.
Converted on an RTX 3080 Ti in 1296.5 s.

**Same layers as the 2.5 build, by construction.** LTX 2.3 and 2.5 share 76 identical 2-D weight
families with identical shapes (read from both headers), and the same profile selects the same
1440 Linear layers on both: every attention and feed-forward projection in the 48 transformer
blocks, video and audio streams and the cross-modal attentions included, plus the audio/video
embeddings connectors. Modulation (`adaln_single`, `scale_shift_table`), norms, patch/caption
projections and `to_gate_logits` are left in BF16.

## Running it

Load it with `CheckpointLoaderSimple` like the factory checkpoint — the file carries the VAEs and
the projection, and ComfyUI reads the quantization metadata the same way it reads the factory
`ltx-2.3-22b-dev-fp8`. Needs `--disable-dynamic-vram`. Text encoder: any Gemma 3 12B in ComfyUI's
single-file format via `LTXAVTextEncoderLoader`, pointed at this file for the projection.

Two things learned while measuring it, both worth more than the checkpoint if you are on a
Windows box with 64 GB of RAM:

- **Keep the encoder out of the render process.** With the 22.7 GB Gemma loaded live beside this
  transformer the same 249-frame run took 638 s; with the conditioning read from a file, 208 s.
  The sampler's eight steps took 18 s either way — the rest was loading and encoding. Encode once,
  render many.
- **Do not save LTX 2.3 conditioning with ComfyUI-LTXVideo's `LTXVSaveConditioning`.** It drops
  the `unprocessed_ltxav_embeds` flag the 2.3 model needs, and the render is noise (measured above).
  The method repo carries a saver that keeps every option (`tools/ltx_encode_lowcommit.py`) and a
  loader node that refuses a file without them (`VoidLoadConditioningFull`). Whatever route you
  use, render the identity control once — the same seed with the encoder live against the saved
  file — before trusting anything rendered on saved conditioning.

## What is NOT covered

- **One prompt, one seed, one resolution (512), one frame rate, one card.** Four arms, not a sweep.
- **MAE, PSNR, SSIM and log-mel distance are distances from the reference, not judgements of
  quality**; no audio metric here was validated against human judgement, and audio-video synchrony
  is not measured. Watch and listen to the files.
- **The text encoder is BF16 in every arm, and it ran once, on a different card.** The prompt was
  encoded on an RTX 3080 Ti and the transformers ran on the RTX 3090; the two cards' encodes differ
  by relative L2 1.0e-3 (87 % of elements bit-equal in bf16), and that footprint — about 2 MAE on
  the frames, 0.07 log-mel — is common to every arm because every arm reads the same file. A W4A8
  build of the same factory Gemma 3 12B (8.31 GiB, 336 layers) **was measured on this exact render
  on 2026-09-14**: its conditioning sits 0.043 rel-L2 from the BF16's (the same on the quantized
  and the dequantized math path), and the W4A8 transformer on that conditioning lands **MAE 6.75 /
  SSIM 0.894 / log-mel 0.087** from the BF16-encoder render — under the transformer's own
  quantization distance (10.39), the same scene, level unchanged. Proofs and the card:
  https://huggingface.co/JoaoZaokk/Gemma-3-12B-it-W4A8-ConvRot. It is not one of the four arms
  above; every arm above reads the BF16 conditioning.
- **The BF16 reference ran through a different loader than the other arms** (GGUF container,
  weights streamed from the page cache) because ComfyUI's safetensors reader cannot open a 39 GiB
  file on this machine. The weights are bit-identical and the Linear outputs were verified equal on
  12 sampled layers; the loader's speed is reported and not compared.
- **No per-layer error analysis for this checkpoint** — no activation calibration was run on it.
- `convrot_groupsize` **256 only**; no mixed build.

## License and changes

LTX 2.3 is distributed under the **LTX-2 Community License Agreement** (January 5, 2026), and this
derivative is distributed **exclusively under those same terms**, as Section 3(b) requires. The
complete agreement ships beside these weights as `LICENSE_LTX_2_COMMUNITY.txt` — read it before use.
Two things in it bind you and anyone you pass this on to: **the use-based restrictions of Section 4
and Attachment A**, and **Section 2**, under which entities with annual revenues of at least
US$10,000,000 need a paid commercial licence from Lightricks to use LTX-2 or any derivative of it,
this file included.

**Statement of changes, per Section 3(c):** the 1440 Linear layers of the transformer were
re-encoded from BF16 into ComfyUI's `asym_w4a8_int8` format (4-bit weights, 8-bit activation path,
Hadamard rotation at group size 256). No weights were fine-tuned, no architecture was altered, and
every other tensor — video VAE, audio VAE, vocoder, text projection, modulation, norms, embeddings —
is preserved from the source byte-for-byte. All copyright and attribution notices of the source are
retained (Section 3(d)).

## Credits

- **[Lightricks](https://huggingface.co/Lightricks/LTX-2.3)** — LTX 2.3 itself and the BF16 weights
  this was converted from.
- **[Comfy-Org](https://huggingface.co/Comfy-Org/ltx-2)** — the single-file Gemma 3 12B text encoder
  used in every arm, and the `ltx-2.3` templates the graph and sigmas were taken from.
- **[city96 / ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** — the loader for the
  third-party Q6_K arm, and the only way the 39 GiB BF16 reference could be loaded on this machine
  at all (read-only memmap, no `torch.empty`).
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format, the `MixedPrecisionOps` dispatch and
  the CUDA kernels that execute it.
- **ComfyUI-MultiGPU (DisTorch2)** — used in four of the six failed reference attempts; it was not
  what failed, and it is what rendered the 2.5 reference arm.
