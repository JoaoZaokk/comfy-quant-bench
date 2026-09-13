---
license: other
license_name: ltx-2.x-community-license
license_link: https://huggingface.co/JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot/blob/main/LICENSE_LTX_2x_COMMUNITY.txt
base_model: Lightricks/LTX-2.5
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

# LTX 2.5 22B distilled — 4-bit weights, a real 10-second video **with its audio**, and the arm that beats it

A `asym_w4a8_int8` build of `ltx-2.5-22b-distilled-transformer`, measured on a **complete
10-second video and its soundtrack** against the BF16 original and against Lightricks' own INT8.

**39.13 GiB → 11.66 GiB, 3.36x lighter**, and **1.95x faster** than the original on the same
249-frame render. It is also **1.90x less faithful in the picture and 2.9x less faithful in the
audio than Lightricks' own INT8 build** — those numbers are here because they are the ones that
decide whether you want this file.

Method, tools and the full measurement log: **https://github.com/JoaoZaokk/comfy-quant-bench**

---

## Correction, 2026-09-13: the first version of this card measured half of what the model makes

LTX 2.5 generates **video and audio in one latent**. The first version of this card decoded only
the video branch, and said so in its "not covered" list as if that were a footnote. It is not: an
audio-video model judged on frames alone is judged on half its output. All three arms were
re-rendered with the audio branch decoded (`LTXVAudioVAEDecode` on the second output of
`LTXVSeparateAVLatent`), and the soundtrack is measured below with the same seed and the same
three transformers.

The re-render also answered a question nobody had asked: **the frames came back pixel-identical**
to the first run in all three arms (mean absolute difference 0.0 on frames 1/63/125/187/249),
across a server restart and, for the BF16 arm, a different disk. The video numbers below are the
first run's, reproduced exactly; the audio numbers are new.

## The measurement: one 10-second video, three transformers, picture and sound

249 frames at 25 fps (**9.96 s** — LTX requires `8n+1`, so 250 is illegal and 249 is the legal
neighbour), 512×512, 3 steps, cfg 1.0, euler, manual sigmas `0.909375, 0.725, 0.421875, 0.0`,
seed 1234, one RTX 3090. Audio: 48 kHz stereo, 9.93 s, decoded by the same audio VAE in every arm.

**The transformer is the only thing that changes.** All three arms use the same encoder
(`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot`), the same video VAE, the same audio VAE, the
same prompt, the same seed and the same sigmas.

### Picture

| arm | GiB | 249 frames | s/frame | MAE vs BF16 | PSNR | SSIM |
|---|---|---|---|---|---|---|
| BF16 original | 39.13 | 780.7 s | 3.14 | — | — | — |
| `comfy-int8-convrot` (Lightricks) | 20.03 | 481.5 s | 1.93 | **4.10** | **29.71 dB** | **0.941** |
| **this file, W4A8** | **11.66** | **400.9 s** | **1.61** | 7.81 | 25.39 dB | 0.895 |

MAE is the mean absolute per-pixel difference on the 0–255 scale, averaged over **all 249
frames**; the spread is tight (ours 7.16–8.84, theirs 3.61–4.61). Motion energy — mean
|frame_t − frame_t−1| — is 1.73 for BF16, 1.78 for INT8, 1.71 for ours: nobody froze and nobody
jittered.

### Sound

| arm | log-mel L1 vs BF16 | SNR | spectral convergence | lag | level |
|---|---|---|---|---|---|
| `comfy-int8-convrot` (Lightricks) | **0.041** | **11.2 dB** | **0.124** | 0.0 ms | −38.8 dBFS |
| **this file, W4A8** | 0.120 | 3.4 dB | 0.311 | 0.0 ms | −38.3 dBFS |
| *control: silence* | 6.980 | 0.0 dB | — | — | — |
| *control: white noise at the reference's RMS* | 1.471 | −3.0 dB | 1.097 | — | — |

The reference soundtrack is ambient — surf, wind, birds — at −38.7 dBFS with 0 % silent
windows. Log-mel L1 is the mean distance between 64-band log-mel spectrograms and is the number
closest to "sounds alike"; SNR is on the raw waveform and is phase-sensitive, which is why the
cross-correlation lag sits beside it (0.0 ms in both arms — nothing slid in time). The two
controls give the scale: silence scores 6.98, white noise at the same loudness scores 1.47.
Both arms are far inside that — INT8 is 36x closer to the original than white noise, ours 12x.
Neither arm changed the level (within 0.4 dB) and neither went mute.

**The audio ranks the arms the same way the picture does, and by a wider margin**: ours is 1.9x
further from the original in the frames and 2.9x further in the soundtrack. On this bench that
is the fifth family in a row where the INT8 arm is the more faithful one — and here the INT8 arm
is Lightricks' own.

![three arms, picture and sound](av/contato_av.png)

Frames 1 / 63 / 125 / 187 / 249, then each arm's log-mel spectrogram and waveform on a shared
scale. The middle row tracks the top row closely — same thin white lighthouse, same rock, same
framing. **The bottom row is a different composition**: a larger, closer, brick-coloured tower with
dark smoke where the original has a light beam. All three are good videos with coherent motion
across the full ten seconds and a soundtrack that belongs to them. **Ours is a good video that is
further from the original, in both senses.** If you want the original's picture and sound, take
Lightricks' INT8 and pay 8.4 GiB and 20 % more time for it. If VRAM is what binds you, this file
gets a real ten seconds of video and audio out of a 24 GB card faster than anything else here.

### Listen for yourself

The proofs ship in this repo, not in a description of them:

| | video + audio (MP4, h264 + AAC) | lossless audio (FLAC) |
|---|---|---|
| BF16 original | `av/bf16_original.mp4` | `av/bf16_original.flac` |
| INT8, Lightricks | `av/int8_lightricks.mp4` | `av/int8_lightricks.flac` |
| **W4A8, this file** | `av/w4a8_this_file.mp4` | `av/w4a8_this_file.flac` |

Every number above was computed on the lossless PNG frames and FLAC files, never on the MP4s;
`av/comparacao_av.json` carries all of them, per arm, including the controls.

## The file

| file | bytes | GiB | layout |
|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-w4a8.safetensors` | 12,520,267,816 | **11.66** | 1440 × `asym_w4a8_int8`, `group_size` 16, `convrot_groupsize` 256 |

Source: `ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 42,018,190,584 B, byte-for-byte the
file `Lightricks/LTX-2.5` publishes. Backend recorded in the sidecar:
`comfy_kitchen.backends.cuda`.

## Running it

Needs `--disable-dynamic-vram`. At 11.66 GiB it is resident on a 24 GB card and needs **no**
block splitting — which is most of why it is faster than the other two arms, both of which have to
move weights every step.

For contrast, and because it is a measured limit rather than a guess: **the BF16 original does not
render this video without help, and one way of helping it kills the server.** Five attempts:

```
BF16, no DisTorch2,   49 frames   -> CUDA error: out of memory (from mem_get_info; the card is exhausted)
BF16, DisTorch2 6 GiB from cuda:1, 49 frames  -> works, 717.8 s
BF16, DisTorch2 6 GiB from cuda:1, 249 frames -> Windows fatal exception: access violation, process dead
BF16, DisTorch2 40 GB on cpu,      249 frames, file on an SMB share, 24 GiB RAM free
                                              -> access violation inside torch/storage.py __getitem__
                                                 while load_torch_file memory-maps the 39 GiB file;
                                                 the ComfyUI process died with it
BF16, DisTorch2 40 GB on cpu,      249 frames, same file on a local NTFS disk, 40 GiB RAM free
                                              -> works, 769.6 s, and pixel-identical to the first run
```

The frame count alone is not the problem — 249 frames run fine on this W4A8 build with no
splitting at all. A 39 GiB model that has to be memory-mapped is what is fragile, and where the
bytes come from matters as much as how many there are.

## What is NOT covered

- **One prompt, one seed, one resolution (512), one frame rate, one card.** Three arms, not a
  sweep.
- **MAE, PSNR, SSIM and log-mel distance are distances from the reference, not judgements of
  quality.** They say this build lands further from the original; they do not say a viewer or a
  listener prefers the original. Watch and listen to the files.
- **No audio metric here was validated against human judgement on this bench**, and
  **audio-video synchrony is not measured** — each branch is compared with its own counterpart.
- **The text encoder is quantized in every arm** (`comfy-int8-convrot`), held fixed so the axis
  stays clean. A BF16-encoder run would measure a different thing and is not included.
- **No per-layer error analysis for this checkpoint.** Unlike the image models on this bench, this
  build predates the calibration record, so its sidecar has no `source_identity_sha256` and there
  is no `err_w4a8` distribution to quote.
- `convrot_groupsize` **256 only**.

## License and changes

LTX 2.5 is distributed under the **LTX-2.x Community License Agreement**, and this derivative is
distributed **exclusively under those same terms**, as Section 3.2 requires. The complete
agreement, including the use restrictions in Section 4 and Attachment A, ships beside these weights
as `LICENSE_LTX_2x_COMMUNITY.txt` — read it before use; its acceptable-use restrictions bind you
and anyone you pass this on to.

**Statement of changes, per Section 3.3:** the 1440 Linear layers of the transformer were
re-encoded from BF16 into ComfyUI's `asym_w4a8_int8` format (4-bit weights, 8-bit activation path,
Hadamard rotation at group size 256). No weights were fine-tuned, no architecture was altered, and
every non-Linear tensor is preserved from the source.

## Credits

- **[Lightricks](https://huggingface.co/Lightricks/LTX-2.5)** — LTX 2.5 itself, the BF16 weights
  this was converted from, and the `comfy-int8-convrot` build used as the comparison arm. That arm
  is more faithful than this one, in picture and in sound, and it is theirs.
- **ComfyUI / comfy-kitchen** — the `asym_w4a8_int8` format, the `MixedPrecisionOps` dispatch and
  the CUDA kernels that execute it.
- **ComfyUI-MultiGPU (DisTorch2)** — without its block splitting the BF16 reference arm could not
  have been rendered at all on this hardware, and there would have been nothing to compare against.
