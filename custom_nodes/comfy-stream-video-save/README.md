# comfy-stream-video-save

One ComfyUI node, **VAE Decode Tiled + Video Combine (streaming)**
(`StreamingTiledDecodeVideoCombine`). It replaces the pair `VAEDecodeTiled` → `VHS_VideoCombine` for video
VAEs (built and measured on LTX-2.3). The output is the same, but the whole video never sits in RAM.

## Why

`VAEDecodeTiled` returns the whole video as one float32 tensor on the CPU. At 361 frames of 1376×1024 that is
6.1 GB per copy. On the machine below, the decode alone took ComfyUI's private memory from 16 GB to 65.5 GB. A
full two-pass LTX render filled 64 GB of RAM and pushed the process into the pagefile.

This node does the arithmetic of `comfy.utils.tiled_scale_multidim`: the same tile positions, feather masks and
summation order. The difference is that time is the outer loop. A run of frames is released as soon as no later
tile can add to it, and each frame goes straight to `VHS_VideoCombine` (the same ffmpeg arguments, metadata and
audio mux), which already consumes frames one at a time. The peak is one temporal window of tiles, about 57
frames with the default `temporal_size` 64 on LTX.

## Measured (RTX 3090, Windows, 64 GB RAM; LTX-2.3 22B W4A8, 1024×1376, 361 frames, one run each)

| | `VAEDecodeTiled` → `VHS_VideoCombine` | this node |
|---|---|---|
| decode only: ComfyUI private-memory peak | 65.5 GB | **13.2 GB** |
| decode only: time | 134 s | **60 s** |
| full two-pass render: private-memory peak | 116.8 GB | **63.6 GB** |
| full two-pass render: time | 18 min 37 s | **13 min 41 s** |

Output: the decoded video is identical (ffmpeg PSNR = inf) and the audio samples hash the same, for both passes.
`test_tiles.py` checks on the CPU that the tiling is bit-identical to `tiled_scale_multidim`:

```
python -m pytest test_tiles.py -q --import-mode=importlib
```

## Use

- Requires [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite). The node calls
  its `VHS_VideoCombine` with `video/h264-mp4`.
- Inputs: the latent, the VAE, the four tiling values of `VAEDecodeTiled`, plus `frame_rate`,
  `filename_prefix`, `pix_fmt`, `crf`, `save_metadata`, `save_output` and optional `audio`.
- Video VAEs only (temporal compression, batch of 1 per latent). VAEs with their own tiling are refused.

## Limits

- Tested on the LTX-2.3 video VAE only. Other video VAEs that go through `decode_tiled_3d` should behave the same,
  but that is not measured.
- h264-mp4 only. Formats with an ffmpeg pre-pass (e.g. gif) need every frame at once, so the streaming does not
  apply to them.

License: GPL-3.0, like ComfyUI, from which the tiling arithmetic is derived.
