"""Decode em tiles + gravacao de video sem montar o video inteiro na RAM.

O caminho comum (VAEDecodeTiled -> VHS_VideoCombine) entrega o video inteiro como um tensor float32 na CPU:
361 quadros de 1376x1024 sao 6,1 GB por copia, e o decode em tiles ainda aloca o divisor dos pesos (2 GB) e o
reshape do no copia tudo de novo. Este no faz a mesma conta do `comfy.utils.tiled_scale_multidim` (mesmas
posicoes, mascaras e ordem de soma, logo o mesmo resultado bit a bit), mas percorre o tempo por fora e solta cada
trecho de quadros assim que nenhum tile futuro pode mais somar nele. Os quadros prontos vao, um por um, para o
proprio `VHS_VideoCombine` (mesmo ffmpeg, metadados e audio), que ja consome os quadros por iteracao.

Pico de RAM: uma janela temporal de tiles (~57 quadros no LTX com temporal_size 64), nao o video.
"""
import logging

import comfy.model_management as mm
import nodes

from .tiles import _escalas, quadros_em_tiles

logger = logging.getLogger("comfy-stream-video-save")


class QuadrosPreguicosos:
    """Sequencia de quadros [H, W, C] que decodifica sob demanda. So o VHS_VideoCombine a ve: ele pede `len`,
    o quadro 0 (miniatura/dimensoes) e depois itera uma vez."""

    def __init__(self, gerar, total):
        self._gerar, self._total = gerar, total
        self._cache0 = None  # primeiro trecho, decodificado para responder images[0] sem decodificar duas vezes

    def __len__(self):
        return self._total

    def _trechos(self):
        if self._cache0 is not None:
            t, self._cache0 = self._cache0, None
            yield t
            yield from self._fonte
        else:
            yield from self._gerar()

    def __getitem__(self, i):
        if i != 0:
            raise IndexError("so o quadro 0 e acessivel diretamente")
        if self._cache0 is None:
            self._fonte = self._gerar()
            self._cache0 = next(self._fonte)
        return self._cache0[0]

    def __iter__(self):
        for trecho in self._trechos():
            yield from trecho  # trecho [t, H, W, C]


class StreamingTiledDecodeVideoCombine:
    """VAEDecodeTiled + VHS_VideoCombine (h264-mp4) sem o video inteiro na RAM."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "step": 1.0}),
                "filename_prefix": ("STRING", {"default": "video/stream"}),
                "pix_fmt": (["yuv420p", "yuv420p10le"],),
                "crf": ("INT", {"default": 19, "min": 0, "max": 100, "step": 1}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "latent/video"
    DESCRIPTION = ("Decodifica o latente de video em tiles e grava com o VHS Video Combine (h264-mp4) sem montar o "
                   "video inteiro na RAM. Mesmo resultado do VAEDecodeTiled com os mesmos parametros.")

    def run(self, samples, vae, tile_size, overlap, temporal_size, temporal_overlap, frame_rate, filename_prefix,
            pix_fmt, crf, save_metadata, save_output, audio=None, prompt=None, extra_pnginfo=None, unique_id=None):
        vhs = nodes.NODE_CLASS_MAPPINGS.get("VHS_VideoCombine")
        if vhs is None:
            raise RuntimeError("VHS_VideoCombine (ComfyUI-VideoHelperSuite) nao esta instalado")
        vae.throw_exception_if_invalid()

        # conversao de parametros igual a nodes.VAEDecodeTiled.decode
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        tc = vae.temporal_compression_decode()
        if tc is None:
            raise RuntimeError("este no e para VAE de video (sem compressao temporal)")
        temporal_size = max(2, temporal_size // tc)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // tc))
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        if latent.ndim != 5:
            raise RuntimeError(f"latente de video 5D esperado, veio {tuple(latent.shape)}")
        comp = vae.spacial_compression_decode()
        tile = (max(2, temporal_size), tile_size // comp, tile_size // comp)
        ov = (max(1, temporal_overlap), overlap // comp, overlap // comp)
        if getattr(vae, "handles_tiling", False):
            raise RuntimeError("VAE com tiling proprio: use o VAEDecodeTiled")

        escala, _ = _escalas(vae.upscale_ratio, vae.upscale_index_formula, 3)
        total = round(escala(0, latent.shape[2])) * latent.shape[0]

        def fn(a):  # igual ao decode_fn de comfy.sd.VAE.decode_tiled_3d
            return vae.first_stage_model.decode(a.to(vae.vae_dtype).to(vae.device)).to(dtype=vae.vae_output_dtype())

        def fin(trecho):
            return vae.process_output(trecho)[0].movedim(0, -1)  # [t, H, W, C]

        def gerar():
            forma = vae._tile_bounded_shape(latent.shape, tile[1], tile[2], tile[0]) \
                if hasattr(vae, "_tile_bounded_shape") else latent.shape
            mm.load_models_gpu([vae.patcher], memory_required=vae.memory_used_decode(forma, vae.vae_dtype),
                               force_full_load=vae.disable_offload)
            for b in range(latent.shape[0]):
                with mm.cuda_device_context(vae.device):
                    yield from quadros_em_tiles(latent[b:b + 1], fn, tile, ov, vae.upscale_ratio, vae.output_channels,
                                                vae.output_device, vae.upscale_index_formula, fin)

        return vhs().combine_video(
            frame_rate=frame_rate, loop_count=0, images=QuadrosPreguicosos(gerar, total),
            filename_prefix=filename_prefix, format="video/h264-mp4", pingpong=False, save_output=save_output,
            prompt=prompt, extra_pnginfo=extra_pnginfo, audio=audio, unique_id=unique_id,
            pix_fmt=pix_fmt, crf=crf, save_metadata=save_metadata, trim_to_audio=False)


NODE_CLASS_MAPPINGS = {"StreamingTiledDecodeVideoCombine": StreamingTiledDecodeVideoCombine}
NODE_DISPLAY_NAME_MAPPINGS = {"StreamingTiledDecodeVideoCombine": "VAE Decode Tiled + Video Combine (streaming)"}
