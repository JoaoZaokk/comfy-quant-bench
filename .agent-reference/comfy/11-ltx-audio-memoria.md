# LTX, áudio, condicionamento e commit

> Referência preservada do CLAUDE.md original, linhas 1026–1166, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## The LTX card measured half the model, and the half it skipped ranks the arms the same way

LTX 2.x generates **video and audio in one latent**. Until 2026-09-13 `tools/ltx25_video.py`
decoded the video branch only — its own docstring said so — and the published LTX 2.5 card compared
three transformers on 249 frames and zero audio samples. The owner called it: *"um gerador de video
como o LTX nao gera somente imagens, ele gera imagem e audio ao mesmo tempo, entao voce tem que
comparar os dois."* `tools/ltx_video.py` now decodes both (`LTXVAudioVAEDecode` on the second
output of `LTXVSeparateAVLatent`) and writes PNGs, a FLAC and an MP4 with the track;
`tools/compara_av.py` measures both branches against a reference with silence and same-RMS white
noise as controls. Re-rendered, same seed, three arms:

```
arm                   MAE   PSNR   SSIM  | log-mel L1   SNR      lag    spec.conv  RMS
int8 Lightricks      4.10  29.71  0.941  |   0.041     11.2 dB   0 ms    0.124   -38.8 dBFS
W4A8 ours            7.81  25.39  0.895  |   0.120      3.4 dB   0 ms    0.311   -38.3 dBFS
control: silence                        |   6.980      0.0 dB
control: white noise, same RMS          |   1.471     -3.0 dB            1.097
```

Audio ranks the arms as the picture does and by a wider margin (1.9x further in frames, 2.9x in
sound); neither arm changed level or slid in time. **The re-render came back pixel-identical to
the first run in all three arms** (MAE 0.0 on frames 1/63/125/187/249), across a server restart
and, for BF16, a different disk — the PNG hashes differ only by the embedded workflow metadata.

**LTX 2.3 distilled 1.1, measured 2026-09-14 on the same protocol at 8 steps, every arm on the
same saved conditioning, reference BF16 through the lossless GGUF container:**

```
arm                            GiB    sampler 8 steps     MAE   PSNR   SSIM  | log-mel  conv   lag    level
BF16 (GGUF, partial load)     39.15   8.05 s/it (66 s)     --     --     --  |   --      --     --   -23.5
GGUF Q6_K (third party)       16.55   5.04 s/it (40 s)   3.59  29.00  0.941 |  0.163  0.419  0 ms   -24.1
W4A8 ours (transformer 11.66) 15.51   2.30 s/it (18 s)  10.39  21.89  0.829 |  0.163  0.424  0 ms   -24.0
W4A4 ours (control)           14.31   1.59 s/it (12 s)  14.45  20.36  0.734 |  0.281  0.491  0 ms   -25.8
controls (log-mel): silence 8.460, white noise at the reference's RMS 1.748
```

W4A8 is a usable 2.3 (P1); W4A4 does not break and is worse (P2 — fourth family that tolerates
A4); the 6-bit GGUF is 2.9x closer in the frames and 2.2x slower per step (P3); **the sound does
not separate W4A8 from Q6_K** (0.163 against 0.163) while the picture puts them 3x apart — on 2.5
the sound ranked the arms with a wider margin than the picture, on 2.3 it ties the first two (P4
not refuted, with a tie). **The tie is the metric saturating, measured the same night:** at 3 steps
(tail of the same schedule, same conditioning) Q6_K stays in phase with the reference (log-mel
0.063, waveform SNR +8.4 dB) and W4A8 does not (0.223, −1.1 dB) — at 8 steps both had drifted out
of phase (SNR −1 dB each) and landed on the same log-mel floor. Caveat: at 3 steps the 2.3 renders a
different, degraded scene in every arm, so that run answers only the mechanism
(`bench/criterio_fechamento_2026-09-14.md`, G). P5 was untestable as written (no DisTorch BF16 arm
ever ran) and P6 refuted (`bench/criterio_ltx23.md`). The "whole run" wall-clock ranked the arms by
which disk they were read from (BF16 GGUF from local C: 155 s; W4A8 from the W: share 208 s) — one
more reason that column is not a speed. Published with MP4/FLAC proofs, the identity control and
the broken-saver negative: **https://huggingface.co/JoaoZaokk/LTX-2.3-22B-distilled-1.1-W4A8-ConvRot**

**And the BF16 arm killed the server once — then the 2.3 work killed it three more times, same
signature.** `Windows fatal exception: access violation` in `torch/storage.py __getitem__` under
`comfy/utils.py:136` (`f.get_tensor(k)` inside `load_torch_file`) — the page-in of a memory-mapped
39–43 GiB safetensors. First from D: with ~24 GiB RAM free; the retry from W: under a second name
(hardlink, same inode) rendered in 769.6 s with 40 GiB free. **This file first recorded W: as "a
local disk". It is not**: `net use` lists `W: \\192.168.3.40\zfe`, an SMB share like D: and P:;
the claim came from a grep of `net use` that only looked for D:. So every BF16 load here was an
mmap over SMB — one survived, four died (32–44 GiB free, no correlation with RAM), while a bare
Python process paged the same 39 GiB file through in 4 s. The redirector was the suspect for a few
hours; the paragraph after this one is the measurement that replaced it. The only local NTFS
volumes are C: and F:. `folder_paths` returns the **first** yaml root that has the name, so yaml order decides which
*volume* a 39 GiB mmap comes from, and no log says which. Proofs on the Hub: `av/*.mp4`,
`av/*.flac`, `av/contato_av.png`, `av/comparacao_av.json`.

**And the redirector was the wrong suspect too — measured 2026-09-14.** The same two-view
mapping that `safe_open` performs, done by hand in a bare Python process without ComfyUI, dies the
same way on the local C: drive as on W:. What was measured on this 39.13 GiB file, with the system
commit counters read before and after each call (`GetPerformanceInfo`):

```
call                                                       commit charge
safetensors.safe_open(framework="pt")                      +40.8 GiB at open, before any tensor is read
   (two copy-on-write views of the same file: memmap2 for the header and
    torch.UntypedStorage.from_file(shared=False) for the data; +80.2 GiB while both are alive)
torch.empty of the model's parameters                      +40.7 GiB more
read-only mmap (what numpy and the GGUF loader use)          0
UntypedStorage.from_file(shared=True)                        0
```

Windows charges a copy-on-write view its whole size at mapping time, so ComfyUI's normal
safetensors path needs twice the file in commit just to open it and three times to build the
model. This machine's commit limit was 124.8 GiB (63.6 GiB of RAM plus a 61 GiB system-managed
pagefile) when this was measured and is **98.72 GiB as of 2026-09-20**, because the pagefile shrank
to 35.07 GiB -- so the trap below is tighter now than the numbers in it suggest. The old figure, for
the record: 124.8 GiB (63.6 GiB of RAM plus a 61 GiB system-managed
pagefile) with about 70 GiB already committed by other processes. When the charge forces the
pagefile to grow, the new view sometimes comes back with the limit raised but the charge not
taken, and the first read through it is an access violation in `torch/storage.py __getitem__` —
the server's exact signature (`safe_open` + first `get_tensor` on W:, 3 of 3 runs; the hand-made
two-view mapping on C:, 1 of 1; the same calls survive on other runs, which is why one BF16 load
in seven succeeded). The two deaths inside `nn.Linear.__init__` — the model's `torch.empty` —
carry the same signature and were not reproduced in isolation. So the list above stays, with a
different reading: rows 3 to 5 died of commit, not of the network. The LTX 2.3 reference was
rendered without the safetensors reader at all: the same BF16 weights, bit for bit, in a GGUF
container read through a read-only memmap (`tools/safetensors_to_gguf_bf16.py`; bytes,
dequantized weight and Linear output verified identical on 12 sampled layers with
`tools/probe_gguf_bf16_equivalence.py`).

**Rule that follows, for this machine:** never open a safetensors bigger than about half the free
commit through ComfyUI's normal reader (`safe_open` costs 2x the file; the model another 1x). For a
BF16 reference of a 20 B+ model, convert it losslessly with `tools/safetensors_to_gguf_bf16.py`
and load it with `UnetLoaderGGUF`; keep the text encoder out of the process with
`tools/ltx_video.py --encode-only` / `--cond-from` (saved conditioning costs nothing). The pagefile
is the owner's, not ours: `?:\pagefile.sys`, system-managed, 61.2 GiB on C: at the time of writing,
and every death above needed it to grow.

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

**"Per-step time depends on residency by 3x" sat in this file for an hour and was wrong — the
field it came from is not per-step time.** `tools/ltx_video.py` writes `s_por_passo` and
`s_por_quadro` as the whole run's wall-clock divided by steps or frames: model load, text-encoder
load and encode, sampling, both VAE decodes and muxing, in one number. The per-step instrument is
the sampler's own tqdm bar in the server log, and it says the same W4A8 249-frame render sampled
**8 steps in 18 s (2.30 s/it) with the encoder live and 8 steps in 18 s (2.30 s/it) with saved
conditioning**. The 638 s against 208 s of wall-clock was loading the 22.7 GB encoder over SMB and
encoding once; residency changed nothing the sampler could see. Same instrument, same protocol,
the other 2.3 arms: W4A4 **1.60 s/it**, GGUF Q6_K **5.12 s/it** (dequantized math), all 249 frames
at 512 px on the 3090. The 2.5 card's "s/frame" column had the same defect — its 3 steps sample in
about 25 s and the runs took 400–800 s — and now says so. **Rule: a speed number from
`ltx_video.py`'s JSON is the wall-clock of a run; a per-step number comes from the progress bar,
or from a tool that times the sampler alone.** The JSON now carries a `nota_tempo` field saying
exactly that.

