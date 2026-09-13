"""Capture the activations a diffusion model really sees, so precision can be chosen per layer.

Every quantization decision in this project so far was made from the weights alone. That answers
half the question: ConvRot W4A4 quantizes the weight *and* the activation, and the activation half
is the one that hurts. Its scheme is one scale per token across every channel, so a single outlier
channel sets the scale for the whole vector -- and which layers get outlier-heavy inputs is a
property of the model running, not of any tensor on disk.

This runs the real model through real sampling steps and keeps, per candidate Linear:

  * `sample`         a reservoir of real input rows, **bfloat16** on CPU -- the tensors the layer
                     was actually handed, not a Gaussian stand-in. It says bf16 because it is
                     bf16: this line said "fp16" for as long as `Reservoir.__init__` stored
                     bf16, and "the sample is fp16" is the exact statement the 344064-vs-65504
                     incident was about. A docstring that still describes the bug is how the bug
                     gets reintroduced downstream -- and it was, in `quant_mixed.py`, which read
                     this and cast the sample back to the checkpoint dtype before measuring.
  * `channel_absmax` running per-channel max|x| over every row seen, not just the sampled ones
  * `crest`          per-token max|x| / rms(x), summarised as mean/p50/p99/max. Kept as a
                     diagnostic, NOT as a predictor: measured across 170 Z-Image layers its
                     rank correlation with the W4A4 error the converter goes on to measure is
                     +0.10, i.e. none. The mechanism is real -- one scale per token means an
                     outlier channel sets the scale for the whole vector -- but it does not
                     reach the output, because the channel that blows up the scale is usually
                     also the channel that dominates the result. `feed_forward.w2` has the
                     highest crest in the model (p50 98, against a theoretical max of
                     sqrt(10240) = 101.2) and `attention.out` has the lowest (p50 16), and
                     `attention.out` holds the second-worst layer. Choose precision by measuring
                     the kernels, not by this number.
  * `calls`/`rows`   how much traffic the layer saw, so a layer that barely ran can be told apart
                     from one that ran constantly

The reservoir is Algorithm R, not "first N rows": the first sampling step of a diffusion model
sees pure noise and looks nothing like the last, and a converter calibrated on step 0 would be
calibrated on the one distribution the model spends the least time in.

Output is a `.calib.pt` consumed by tools/quant_mixed.py. It does not decide anything itself --
deciding needs the weights and the kernels, and that belongs in the converter.

    python_embeded\\python.exe -s tools/calibrate_activations.py \\
        --model beyond-reality-zimage-v2_bf16.safetensors --profile zimage \\
        --clip qwen_3_4b.safetensors --clip-type lumina2 \\
        --prompt-file tools/benchmark_prompt.txt --steps 8 --seeds 1234 5678 \\
        --out calib/zimage_v2.calib.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
import zlib
from pathlib import Path

PORTABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORTABLE_ROOT / "ComfyUI"))

import torch  # noqa: E402

# Safetensors header size cap from the format's own spec. A .ckpt or a truncated download read
# through this function otherwise asks for an arbitrary allocation from the first eight bytes.
MAX_SAFETENSORS_HEADER = 100_000_000


# Body sampling for the identity digest. Eight chunks rather than the two obvious ones (head and
# tail) because head+tail alone still could not separate two of the colliding pairs measured
# below; eight separates all 45 checkpoints on this bench. 2 MiB total, eight seeks, O(1) in model
# size -- the cost is the same for a 400 MiB VAE and a 14 GiB transformer.
IDENTITY_SAMPLE_CHUNKS = 8
IDENTITY_SAMPLE_BYTES = 256 * 1024


def safetensors_identity_digest(path: Path) -> str:
    """sha256 over the safetensors header plus a bounded sample of the body.

    Lives here, next to the profile tables, because `quant_mixed.py` already imports this module
    and the alternative is the same digest defined twice -- which is how two definitions drift and
    stop agreeing about what "the same checkpoint" means.

    ## What it distinguishes, and what it does not

    **It is a sample, not a hash of the file.** Two checkpoints that agree on the header and on
    all eight sampled windows collide, and nothing here would notice. It is chosen to be strong
    enough for the question actually asked -- *is this analysis about this checkpoint?* -- at a
    cost that does not scale with a 14 GiB model. A real content identity means reading the whole
    file, which this bench cannot afford per invocation and which `tools/model_audit.py` already
    provides for the cases that need it.

    ## Why the body is sampled at all -- MEASURED 2026-08-22, not argued

    Provenance was originally checked by BASENAME. That was replaced with a sha256 of the raw
    header alone, on the stated reasoning that "the header alone identifies a checkpoint -- every
    tensor name, dtype, shape and offset is in it". Every clause of that is true and the
    conclusion does not follow: the header describes the **layout**, and two models of the same
    architecture have identical layouts and different weights.

    Over the 45 `.safetensors` in `ComfyUI/models/diffusion_models`, `ComfyUI/models/unet` and
    `D:/ComfyUI-Models/diffusion_models`, header-only produced **four colliding groups covering
    nine files**, each group also identical in byte size:

        e36743d8e80c9cef  453 keys   beyond-reality-zimage-v2_native / z_image_de_turbo_v1_bf16
                                     / z_image_turbo_bf16
        bced9dae1b9a4c0c  521 keys   beyond-reality-zimage-v2_bf16 / beyond-reality-recovered-bf16
        c78d89e7215f3a98 1024 keys   void_pass2 / void_pass1
        a904e26816259491 1908 keys   wan2.2_i2v_high_noise_14B_fp8_scaled / ..._low_noise_...

    The first group is *the three checkpoints `--foreign-analysis`'s own help text names as
    different models*. The third is two passes of one conversion. The fourth is the two halves of
    a Wan i2v pair. For every one of these the basename differs, so the check being replaced
    would have **refused** the pairing and the header-only digest **accepted** it -- the guard
    became weaker than the one it replaced, on exactly the model family this bench converts.

    With the body sampled: 45 distinct digests, zero collisions. Re-run the measurement with
    `tools/probe_backend_resolution.py`'s sibling, or by hashing the two ways and grouping.

    ## Why a basename check still runs alongside this

    See `quant_mixed.py`'s `foreign` computation. A false accept here assigns per-layer precision
    from the wrong model's activations and produces a file that looks fine; a false refuse costs a
    flag. The two checks are therefore ORed, not swapped.
    """
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise SystemExit(f"{path}: too short to be a safetensors file")
        size = struct.unpack("<Q", raw_size)[0]
        if size == 0 or size > MAX_SAFETENSORS_HEADER:
            raise SystemExit(
                f"{path}: safetensors header length reads as {size} bytes, which is not a "
                "safetensors header. Is this file a .ckpt, or truncated?")
        header = handle.read(size)
        if len(header) != size:
            raise SystemExit(
                f"{path}: header claims {size} bytes, only {len(header)} are present")
    # Header first, then the sample. Hashing the header alone is what this function used to do
    # and is preserved as the prefix, so a layout difference still shows up even if the sampled
    # windows happen to agree.
    digest = hashlib.sha256(header)
    body_start = 8 + size
    total = path.stat().st_size
    span = total - body_start
    if span >= IDENTITY_SAMPLE_BYTES:
        with path.open("rb") as handle:
            last = IDENTITY_SAMPLE_CHUNKS - 1
            for index in range(IDENTITY_SAMPLE_CHUNKS):
                offset = body_start + (span - IDENTITY_SAMPLE_BYTES) * index // max(last, 1)
                handle.seek(max(body_start, min(offset, total - IDENTITY_SAMPLE_BYTES)))
                digest.update(handle.read(IDENTITY_SAMPLE_BYTES))
    return digest.hexdigest()

# Kept identical to the converter's, and imported by it, so a layer can never be calibrated under
# one definition and quantized under another.
PROFILE_PATTERNS = {
    # Z-Image / Lumina-NextDiT, in ComfyUI's *native* naming -- `attention.qkv` (q, k and v fused
    # into one [3*dim, dim] Linear) and `attention.out`, not the checkpoint's diffusers-style
    # `to_q`/`to_k`/`to_v`/`to_out.0`. A published Z-Image checkpoint is in diffusers naming and
    # must go through tools/to_native.py first; quantizing the diffusers names produces a file
    # ComfyUI silently loads without its scales (see that tool's docstring).
    # 34 blocks (30 `layers` + 2 of each refiner) x 5 Linears = 170.
    # Deliberately excluded: `adaLN_modulation`, every norm, `cap_embedder`, `x_embedder` and
    # `final_layer` -- the patchify/unpatchify pair and the modulation path, which is the "keep
    # the start and the end higher" rule this project settled on.
    "zimage": re.compile(
        r"^(?:layers|noise_refiner|context_refiner)\.\d+\."
        r"(?:attention\.(?:qkv|out)|feed_forward\.w[123])$"
    ),
    "ltx_2_5": re.compile(
        r"^(?:model\.diffusion_model\.)?"
        r"(?:"
        r"transformer_blocks\.\d+\."
        r"(?:"
        r"(?:audio_)?attn\d+\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_to_video|video_to_audio)_attn\.(?:to_[qkv]|to_out\.\d+)"
        r"|(?:audio_)?ff\.net\.\d+(?:\.proj)?"
        r")"
        r"|(?:audio|video)_embeddings_connector\.transformer_\d+d_blocks\.\d+\."
        r"(?:attn\d+\.(?:to_[qkv]|to_out\.\d+)|ff\.net\.\d+(?:\.proj)?)"
        r")$"
    ),
    # HunyuanVideo 1.5, 54 `double_blocks` x 8 Linears = 432. Corrected 2026-08-19 on first
    # execution: this pattern read `img_attn_qkv` and `img_mlp.fc1`, and the model's modules are
    # `img_attn.qkv` and `img_mlp.0` -- a dot where it expected an underscore, and numbered
    # Sequential entries where it expected named ones. It matched **nothing**, and
    # `calibrate_activations` said so loudly ("matched no Linear in HunyuanVideo") rather than
    # calibrating an empty set, which is the only reason this was cheap to find. The profile had
    # been written from the checkpoint's key names without ever being run against a loaded model.
    # Excluded on purpose, same rule as zimage: `img_mod.lin` / `txt_mod.lin` are the modulation
    # path, and `txt_in`, `vision_in`, `byt5_in`, `time_in` and `final_layer` are the ends.
    "hunyuan_video_15": re.compile(
        r"^double_blocks\.\d+\.(?:(?:img|txt)_attn\.(?:qkv|proj)|(?:img|txt)_mlp\.[02])$"
    ),
    # WAN 2.1, 30 `blocks` x 10 Linears = 300. `vace_blocks` (the VACE control branch) is left
    # out: it only runs when a control input is present, so a calibration would record nothing for
    # it and the converter would then have to guess. Excluded as ever: patch/vace embeddings,
    # `norm_q`/`norm_k`, `time_embedding`, `text_embedding` and the head.
    "wan_2_1": re.compile(
        r"^blocks\.\d+\.(?:(?:self|cross)_attn\.[qkvo]|ffn\.[02])$"
    ),
    # FLUX.1 (`double_blocks` + `single_blocks`). 19 double x 10 Linears = 190, 38 single x 3 =
    # 114, total **304** dos 314 modulos 2-D do arquivo. Os 10 de fora sao exatamente as pontas
    # que todo perfil aqui exclui: `final_layer`, `guidance_in`, `img_in`, `time_in`, `txt_in`,
    # `vector_in`. Todos os 304 tem a coluna divisivel por 256.
    #
    # DERIVADO, NAO CHUTADO, mas por um caminho diferente do `qwen_image` e do `wan_2_2`. Nao ha
    # `int8_convrot` publico do FLUX.1 para conferir contra -- os `wraps/FLUX.2-klein-*` sao
    # outra arquitetura (klein, sem `double_blocks`/`single_blocks`) e derivar dali seria
    # invalido. O segundo lado veio do CODIGO do ComfyUI: `comfy/ldm/flux/layers.py` constroi
    # `self.qkv`/`self.proj` dentro de `SelfAttention`, `self.linear1`/`linear2` no
    # `SingleStreamBlock` e `self.modulation` com `self.lin` -- entao os nomes de MODULO batem
    # com as chaves do ARQUIVO. Isso importa: o perfil `hunyuan_video_15` logo acima foi escrito
    # das chaves do arquivo, os nomes de modulo eram outros, e ele casou NADA num modelo
    # carregado.
    #
    # AINDA NAO CONFERIDO CONTRA UM MODELO CARREGADO. Escrito 2026-09-13 com a GPU ocupada; a
    # checagem por `named_modules()` e o primeiro passo antes de qualquer conversao usar isto.
    #
    # `img_mod.lin`, `txt_mod.lin` e `single_blocks.N.modulation.lin` ENTRAM, ao contrario do
    # `zimage` e do `hunyuan_video_15`. Nao e inconsistencia: esta bancada ja mediu em QUATRO
    # arquiteturas que a modulacao e a camada MAIS BARATA do bloco em 4 bits -- no Qwen Edit,
    # `txt_mod.1` da 0,0248 contra 0,2101 da saida da atencao, fator 8,5.
    "flux_1": re.compile(
        r"^(?:double_blocks\.\d+\.(?:(?:img|txt)_attn\.(?:qkv|proj)|(?:img|txt)_mlp\.[02]"
        r"|(?:img|txt)_mod\.lin)|single_blocks\.\d+\.(?:linear[12]|modulation\.lin))$"
    ),
    # WAN 2.2. O `ti2v_5B` e IDENTICO ao 2.1 em nomes -- 30 blocks x 10 = 300, e o perfil acima
    # ja serve. O `animate_14B` NAO: ele tem 40 blocks e mais duas familias por bloco,
    # `cross_attn.k_img` e `cross_attn.v_img`, as projecoes de cross-attention de IMAGEM, que nao
    # existem num T2V puro e por isso nunca precisaram entrar no perfil do 2.1.
    #
    # NAO adivinhado: conferido contra o `wan2.2_animate_14B_int8_convrot` do proprio Comfy-Org,
    # que carrega 480 tensores `.comfy_quant`. O perfil do 2.1 casava 400 -- subconjunto ESTRITO,
    # zero camadas nossas fora da lista deles -- e as 80 que faltavam sao exatamente essas duas
    # familias. Conferido pelos dois lados, como o `qwen_image`.
    "wan_2_2": re.compile(
        r"^blocks\.\d+\.(?:(?:self|cross)_attn\.(?:[qkvo]|[kv]_img)|ffn\.[02])$"
    ),
    # Qwen-Image / Qwen-Image-Edit, MMDiT com duas correntes, 60 `transformer_blocks` x 14
    # Linears = 840. NAO adivinhado: derivado do `qwen_image_edit_2511_int8_convrot` do
    # proprio Comfy-Org, cujas 840 chaves `comfy_quant` sao exatamente estas -- sem falso
    # positivo e sem falso negativo -- e conferido pelos DOIS lados, porque a licao do
    # `hunyuan_video_15` acima foi um perfil escrito das chaves do arquivo que casou NADA no
    # modelo carregado. Carregado de verdade na 3090: `named_modules()` devolve estes mesmos
    # 840 nomes.
    #
    # `img_mod.1` e `txt_mod.1` sao MODULACAO e entram, ao contrario do `zimage` e do
    # `hunyuan_video_15` aqui em cima. Nao e inconsistencia minha: o Comfy-Org os quantiza, e
    # o folclore "modulacao nunca" ja foi medido e refutado nesta bancada -- em W4A4 o
    # `adaLN_modulation` do Z-Image e a camada de MENOR erro do bloco inteiro.
    #
    # FORA, e sao exatamente as pontas, os 6 tensores 2-D que sobram do arquivo: `img_in`,
    # `txt_in`, `proj_out`, `norm_out.linear` e o par `time_text_embed.timestep_embedder.*`.
    "qwen_image": re.compile(
        r"^transformer_blocks\.\d+\."
        r"(?:attn\.(?:to_[qkv]|to_out\.0|add_[qkv]_proj|to_add_out)"
        r"|(?:img|txt)_mlp\.net\.(?:0\.proj|2)"
        r"|(?:img|txt)_mod\.1)$"
    ),
    # Krea2 / SingleStreamDiT (`comfy/ldm/krea2/model.py`), 28 `blocks` x 8 Linears = 224.
    # O nome no MODULO e o nome no ARQUIVO sao a mesma string aqui -- o checkpoint ja vem com
    # `blocks.N.attn.wq.weight`, sem prefixo e sem fusao de qkv -- entao nao ha entrada em
    # MODULE_TO_FILE. Conferido pelos dois lados antes de rodar: o regex casa 224 pesos no
    # header do BF16 e o `krea2_turbo_int8_convrot` publico carrega 224 tensores `comfy_quant`.
    #
    # `attn.gate` e uma QUINTA Linear da atencao (`model.py:72`), aplicada como
    # `wo(out * sigmoid(gate))` -- nao e modulacao e nao e norma, e sozinha ela e 8,7% dos
    # parametros do modelo. Deixar de fora por parecer "portao" custaria mais do que quantizar
    # qualquer bloco inteiro.
    #
    # FORA, pela regra "as pontas ficam mais altas": `txtfusion.*` (4 blocos, 32 Linears, 2,7%
    # dos parametros) e a ENTRADA do condicionamento de texto; `first` / `last.linear` sao o
    # patchify/unpatchify; `tmlp` / `tproj` / `txtmlp` sao embeddings. E `blocks.N.mod.lin` nao
    # aparece aqui porque NAO E Linear: e `nn.Parameter` (`model.py:110`, `:121`), entao o
    # caminho de modulacao esta fora por construcao, nao por escolha.
    "krea2": re.compile(
        r"^blocks\.\d+\.(?:attn\.(?:w[qkvo]|gate)|mlp\.(?:gate|up|down))$"
    ),
}

# The patterns above match **module** names, because that is what this file hooks. Downstream,
# `quant_mixed.py` matches **checkpoint key** names, because that is what it rewrites. For Z-Image
# and LTX those are the same string and nothing forced the distinction into the open. For
# HunyuanVideo 1.5 they are not:
#
#     file    double_blocks.0.img_attn_qkv.weight     double_blocks.0.img_mlp.fc1.weight
#     module  double_blocks.0.img_attn.qkv            double_blocks.0.img_mlp.0
#
# ComfyUI renames on load. A calibration keyed by module name would therefore describe layers that
# `quant_mixed` cannot find, and the merge would come back empty -- or worse, partially populated.
# So the calibration is written out keyed by the **file** name, and the translation lives here,
# next to the pattern it belongs to.
MODULE_TO_FILE = {
    "hunyuan_video_15": (
        (re.compile(r"\.(img|txt)_attn\.(qkv|proj)$"), r".\1_attn_\2"),
        (re.compile(r"\.(img|txt)_mlp\.0$"), r".\1_mlp.fc1"),
        (re.compile(r"\.(img|txt)_mlp\.2$"), r".\1_mlp.fc2"),
    ),
    # WAN keeps the module names and prefixes them. Third distinct convention in this table, and
    # the third time the file and the module disagreed -- worth stating plainly: assume they
    # differ until a dump of both says otherwise.
    "wan_2_1": (
        (re.compile(r"^blocks\."), "model.diffusion_model.blocks."),
    ),
}


def to_file_name(profile: str, module_name: str) -> str:
    """Module name -> checkpoint key stem, for profiles where ComfyUI renames on load."""
    for pattern, replacement in MODULE_TO_FILE.get(profile, ()):
        module_name = pattern.sub(replacement, module_name)
    return module_name


# What `quant_mixed.py` matches against checkpoint keys. Only profiles whose file naming differs
# from their module naming need an entry; the rest fall back to PROFILE_PATTERNS.
PROFILE_FILE_PATTERNS = dict(PROFILE_PATTERNS)
PROFILE_FILE_PATTERNS["hunyuan_video_15"] = re.compile(
    r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])$"
)
PROFILE_FILE_PATTERNS["wan_2_1"] = re.compile(
    r"^(?:model\.diffusion_model\.)?blocks\.\d+\.(?:(?:self|cross)_attn\.[qkvo]|ffn\.[02])$"
)
PROFILE_FILE_PATTERNS["wan_2_2"] = re.compile(
    r"^(?:model\.diffusion_model\.)?blocks\.\d+\."
    r"(?:(?:self|cross)_attn\.(?:[qkvo]|[kv]_img)|ffn\.[02])$"
)


class Reservoir:
    """Algorithm R over token rows, so the sample represents the whole run and not its start.

    Vectorised, because the obvious per-row loop is not viable here: one Z-Image call hands a
    layer ~4100 token rows, and 238 layers over 8 steps and 4 runs is 31 million rows. Instead
    every row in a call draws its slot at once, and only the rows that survive -- after collapsing
    duplicate slots to the last writer, which is what the sequential algorithm would have left --
    cross to the CPU buffer.
    """

    def __init__(self, capacity: int, width: int, seed: int):
        self.capacity = capacity
        # Sigma da chamada que produziu cada linha guardada, na MESMA posicao do buffer.
        # Existe porque o dano da quantizacao nao e uniforme ao longo da trajetoria: medido em
        # 2026-08-30 com `tools/probe_epsilon_per_step.py`, o erro do epsilon vai de 6,17e-1 em
        # sigma 1,000 para 5,31e-2 em 0,300, monotonico. Um criterio que pese as linhas por
        # onde elas foram observadas e uma coisa diferente de um que trate todo passo igual, e
        # sem esta coluna essa pergunta nao pode nem ser feita: o reservoir mistura os passos
        # por construcao (Algoritmo R sobre o fluxo inteiro) e depois nao da para separar.
        # NaN aqui significa "nao rotulada" -- o `--sigma-weight` do quant_mixed recusa, em vez
        # de tratar como zero, que seria descartar a linha em silencio.
        self.sigma = torch.full((capacity,), float("nan"), dtype=torch.float32)
        # bfloat16, not float16, for the same two bytes. Measured on Z-Image:
        # `layers.0.feed_forward.w2` receives channel magnitudes up to 344064, which overflows
        # fp16's 65504 to inf. Stored that way, every error metric for that layer came back nan,
        # `nan > threshold` is False, and the layer with the largest activations in the model was
        # assigned the *cheapest* format. bf16 carries fp32's exponent range, so it cannot.
        self.buffer = torch.zeros(capacity, width, dtype=torch.bfloat16)
        self.filled = 0
        self.seen = 0
        self.generator = torch.Generator().manual_seed(seed)

    def offer(self, rows: torch.Tensor, sigma: float = float("nan")) -> None:
        n = rows.shape[0]
        if n == 0:
            return
        if self.filled < self.capacity:
            take = min(self.capacity - self.filled, n)
            self.buffer[self.filled:self.filled + take] = rows[:take].to(
                device="cpu", dtype=torch.bfloat16)
            self.sigma[self.filled:self.filled + take] = sigma
            self.filled += take
            self.seen += take
            rows = rows[take:]
            n = rows.shape[0]
            if n == 0:
                return

        # Row i (1-based over the whole stream) draws a slot uniform in [0, i) and is kept only
        # when that lands inside the reservoir. float64 because `seen` passes 2**24 within a
        # single run, where float32 can no longer represent consecutive integers.
        index = torch.arange(self.seen + 1, self.seen + n + 1, dtype=torch.float64)
        draw = torch.rand(n, generator=self.generator, dtype=torch.float64)
        slots = (draw * index).floor().to(torch.long)
        self.seen += n

        keep = (slots < self.capacity).nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            return
        chosen = slots[keep]
        # Stable sort by slot, then take the last entry of each run: that is the row the
        # sequential algorithm would have left in the slot, and unlike index_put_ with duplicate
        # indices it is deterministic.
        order = torch.argsort(chosen, stable=True)
        chosen, keep = chosen[order], keep[order]
        last = torch.ones_like(chosen, dtype=torch.bool)
        last[:-1] = chosen[1:] != chosen[:-1]
        self.buffer[chosen[last]] = rows[keep[last]].to(device="cpu", dtype=torch.bfloat16)
        # Mesmos slots, mesma mascara: a linha e o sigma dela tem de andar juntos ou a coluna
        # mente sobre de onde a linha veio, que e pior do que nao existir.
        self.sigma[chosen[last]] = sigma


class LayerStats:
    """Per-layer accumulators. The *statistics* stay on the GPU until finish().

    Pulling a scalar to the CPU inside a forward hook forces a device sync on every one of the
    ~7600 calls a run makes, so `channel_absmax` and the crest chunks are accumulated on device
    and moved once, at the end.

    The reservoir is a different matter and this docstring used to overstate it: `offer()` writes
    into a CPU buffer, so it syncs on any call that accepts at least one row. With the real
    calibration numbers (`layers.0`, 159744 rows over 32 calls, capacity 128) the chance a given
    late call accepts nothing is `(154752/159744)**128 = 0.017`, i.e. it syncs on ~98% of calls.
    That is inherent to keeping the sample on the host; what was removed is the *avoidable* sync,
    not all of it.
    """

    def __init__(self, width: int, capacity: int, seed: int, device: torch.device):
        self.reservoir = Reservoir(capacity, width, seed)
        self.channel_absmax = torch.zeros(width, dtype=torch.float32, device=device)
        self.crest_chunks = []
        self.calls = 0
        self.rows = 0

    def observe(self, x: torch.Tensor, crest_rows: int,
                sigma: float = float("nan")) -> None:
        flat = x.reshape(-1, x.shape[-1])
        self.calls += 1
        self.rows += flat.shape[0]

        magnitude = flat.abs().float()
        torch.maximum(self.channel_absmax, magnitude.amax(dim=0), out=self.channel_absmax)

        # Crest factor is what the activation quantizer's one-scale-per-token design is sensitive
        # to. Capped rows per call because it is a diagnostic, not the decision.
        head = magnitude[:crest_rows]
        rms = head.pow(2).mean(dim=1).sqrt().clamp(min=1e-12)
        self.crest_chunks.append(head.amax(dim=1) / rms)

        self.reservoir.offer(flat, sigma)

    def finish(self) -> dict:
        crest = (torch.cat(self.crest_chunks).cpu() if self.crest_chunks
                 else torch.zeros(1))
        quantiles = torch.quantile(crest.float(), torch.tensor([0.5, 0.99]))
        sample_sigma = self.reservoir.sigma[:self.reservoir.filled].clone()
        return {
            "sample": self.reservoir.buffer[:self.reservoir.filled].clone(),
            "sample_sigma": sample_sigma,
            "sigma_tagged": bool(sample_sigma.numel()) and bool(torch.isfinite(sample_sigma).all()),
            "channel_absmax": self.channel_absmax.cpu(),
            "crest_mean": float(crest.mean()),
            "crest_p50": float(quantiles[0]),
            "crest_p99": float(quantiles[1]),
            "crest_max": float(crest.max()),
            "calls": self.calls,
            "rows": self.rows,
            "sampled_from": self.reservoir.seen,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="diffusion model name or path; must be the high-precision source")
    parser.add_argument("--profile", choices=list(PROFILE_PATTERNS), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip", nargs="+", required=True)
    parser.add_argument("--clip-type", default="lumina2")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", action="append", default=[], type=Path,
                        help="file read as one prompt; repeatable")
    parser.add_argument("--negative", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234],
                        help="one full sampling run per seed")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=1,
                        help="frames for a video model; ignored when the latent format is 2-D")
    parser.add_argument("--rows", type=int, default=128,
                        help="reservoir rows kept per layer")
    parser.add_argument("--crest-rows", type=int, default=64,
                        help="rows per call used for the crest-factor diagnostic")
    parser.add_argument("--attention", choices=["default", "sage"], default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing calibration: {args.out}")
    # UMA LINHA = UM PROMPT. Ate 2026-09-12 esta funcao fazia `p.read_text().strip()`, colando o
    # arquivo INTEIRO num unico prompt: `bench/prompts_zimage_runtime.txt`, com seis prompts,
    # virava um prompt de seis linhas e o cabecalho anunciava "encoding 1 prompt(s)". Nao falhava
    # -- calibrava, e calibrava sobre ativacoes de UM condicionamento em vez de seis, que e pior
    # que falhar porque o numero sai e parece bom. O mesmo defeito estava em quality_ladder.py e
    # foi corrigido la no mesmo dia; este arquivo era o irmao que ficou para tras.
    prompts = list(args.prompt)
    for caminho in args.prompt_file:
        linhas = [ln.strip() for ln in caminho.read_text(encoding="utf-8").splitlines()]
        prompts.extend(ln for ln in linhas if ln and not ln.startswith("#"))
    if not prompts:
        raise SystemExit("Give at least one --prompt or --prompt-file")

    # Same ordering constraint as tools/nunchaku_compare.py: the attention backend is chosen when
    # comfy.ldm.modules.attention is first imported, and every block binds the chosen function
    # into its own namespace at that moment. Setting the flag later patches a name nobody reads.
    import comfy.cli_args
    if args.attention == "sage":
        comfy.cli_args.args.use_sage_attention = True

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import folder_paths

    extra = PORTABLE_ROOT / "ComfyUI" / "extra_model_paths.yaml"
    if extra.is_file():
        import utils.extra_config
        utils.extra_config.load_extra_path_config(str(extra))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; calibration must run where the model runs")

    # HIGH_VRAM keeps the whole transformer resident, which is what makes the hooks cheap -- but
    # it is a promise the card cannot always keep. LTX 2.5 is 39 GiB of BF16 against 24 GiB of
    # RTX 3090, and forcing HIGH_VRAM there turns "slow" into "out of memory". Decided from the
    # file size against the device, not assumed.
    candidate_size = Path(args.model)
    if not candidate_size.is_file():
        import folder_paths as _fp
        candidate_size = Path(_fp.get_full_path_or_raise("diffusion_models", args.model))
    model_gib = candidate_size.stat().st_size / 2 ** 30
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if model_gib < 0.7 * total_gib:
        comfy.model_management.vram_state = comfy.model_management.VRAMState.HIGH_VRAM
        comfy.model_management.set_vram_to = comfy.model_management.VRAMState.HIGH_VRAM
    else:
        print(f"{model_gib:.1f} GiB model against {total_gib:.1f} GiB of VRAM: leaving ComfyUI's "
              f"own VRAM policy alone. Calibration will offload and be slow; the numbers are the "
              f"same, the wall clock is not.", flush=True)

    # Encode every prompt and drop the encoder before the transformer is loaded. Qwen3-4B is
    # ~8 GiB and a 12 GiB BF16 transformer alongside it does not fit on a 24 GiB card.
    clip_paths = [folder_paths.get_full_path_or_raise("text_encoders", c) for c in args.clip]
    print(f"encoding {len(prompts)} prompt(s) with {[Path(c).name for c in clip_paths]}",
          flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=clip_paths,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, args.clip_type.upper()))
    conditioning = []
    for text in prompts:
        positive = [list(clip.encode_from_tokens_scheduled(clip.tokenize(text))[0])]
        negative = [list(clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))[0])]
        positive[0][0] = positive[0][0].clone().cpu()
        negative[0][0] = negative[0][0].clone().cpu()
        conditioning.append((positive, negative))
        print(f"  {list(positive[0][0].shape)}  |cond| {float(positive[0][0].float().norm()):.4f}",
              flush=True)
    del clip
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()

    candidate = Path(args.model)
    path = str(candidate) if candidate.is_file() else \
        folder_paths.get_full_path_or_raise("diffusion_models", args.model)
    # Taken before the model loads, so a file that cannot be identified costs a seek rather than
    # a five-minute sampling run. This is what `quant_mixed.py` refuses on; the basename it used
    # to compare is kept in `meta["source"]` for humans only.
    source_digest = safetensors_identity_digest(Path(path))
    print(f"loading {path}", flush=True)
    model = comfy.sd.load_diffusion_model(path)
    diffusion_model = model.get_model_object("diffusion_model")

    # A quantized checkpoint would calibrate the damage instead of the signal. Refuse rather than
    # produce a plausible-looking file that describes the wrong thing.
    #
    # This used to test `named_buffers()` for a `comfy_quant` suffix, and that check could never
    # fire. Measured 2026-08-19 on the mixed Z-Image checkpoint: 170 `comfy_quant` keys in
    # `state_dict()`, 170 modules whose `.weight` is a QuantizedTensor, **0** in `named_buffers()`
    # and 0 in `named_parameters()`. The marker is materialised by the quantized Linear, not
    # registered as a buffer. So the guard that was supposed to stop a calibration from running on
    # an already-quantized file had never stopped anything -- it had simply never been given one.
    from comfy_kitchen.tensor.base import QuantizedTensor
    already = [name for name, module in diffusion_model.named_modules()
               if isinstance(getattr(module, "weight", None), QuantizedTensor)]
    if already:
        raise SystemExit(f"{path} has {len(already)} QuantizedTensor weights; calibration needs "
                         "the high-precision source")

    pattern = PROFILE_PATTERNS[args.profile]
    targets = {name: module for name, module in diffusion_model.named_modules()
               if isinstance(module, torch.nn.Linear) and pattern.match(name)}
    if not targets:
        raise SystemExit(f"Profile {args.profile!r} matched no Linear in "
                         f"{type(diffusion_model).__name__}")
    print(f"hooking {len(targets)} Linear layers matching profile {args.profile!r}", flush=True)

    stats: dict[str, LayerStats] = {}
    handles = []

    # O hook e um forward-pre-hook numa Linear: ele nao ve o timestep. Quem ve e o wrapper de
    # `apply_model`, e todo hook que dispara dentro de uma chamada dele compartilha o mesmo
    # sigma -- e a mesma tecnica que `tools/probe_epsilon_per_step.py` usa para casar entradas.
    # Uma lista de um elemento, e nao `nonlocal`, porque `make_hook` fecha sobre o objeto.
    current_sigma = [float("nan")]

    def sigma_wrapper(apply_model, kwargs):
        t = kwargs["timestep"]
        try:
            current_sigma[0] = float(t.detach().reshape(-1)[0])
        except Exception:
            current_sigma[0] = float("nan")
        return apply_model(kwargs["input"], t, **kwargs["c"])

    model.model_options = dict(model.model_options)
    if model.model_options.get("model_function_wrapper") is not None:
        raise SystemExit(
            "model_function_wrapper ja esta ocupado. Encadear em silencio trocaria o que o "
            "outro wrapper faz; parar aqui e a opcao honesta.")
    model.model_options["model_function_wrapper"] = sigma_wrapper

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            entry = stats.get(name)
            if entry is None:
                # Seeded per layer name so a rerun with the same arguments samples the same rows,
                # which is what makes two calibrations comparable. crc32 rather than hash():
                # Python randomises string hashing per process, so hash() would silently make
                # every run pick different rows while this comment claimed otherwise.
                entry = LayerStats(x.shape[-1], args.rows,
                                   seed=zlib.crc32(name.encode("utf-8")),
                                   device=x.device)
                stats[name] = entry
            with torch.no_grad():
                entry.observe(x.detach(), args.crest_rows, current_sigma[0])
        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    latent_format = model.model.latent_format
    side = max(args.size // 8, 8)
    # `latent_dimensions` is 2 for image models and 3 for video ones, and a video model handed a
    # 4-D latent does not fail cleanly -- it fails somewhere inside the transformer with a shape
    # error that says nothing about the latent. Read the format rather than assuming images.
    dimensions = getattr(latent_format, "latent_dimensions", 2)
    if dimensions == 3:
        ratio = getattr(latent_format, "temporal_downscale_ratio", 4)
        frames = max(1, (args.frames - 1) // ratio + 1)
        shape = [1, latent_format.latent_channels, frames, side, side]
        print(f"video latent {shape} ({args.frames} frames / temporal ratio {ratio})", flush=True)
    else:
        shape = [1, latent_format.latent_channels, side, side]
    started = time.perf_counter()
    runs = 0
    try:
        for prompt_index, (positive, negative) in enumerate(conditioning):
            for seed in args.seeds:
                latent = torch.zeros(shape, device="cpu")
                noise = comfy.sample.prepare_noise(latent, seed, None)
                print(f"run {runs + 1}/{len(conditioning) * len(args.seeds)} "
                      f"prompt {prompt_index} seed {seed} steps {args.steps}", flush=True)
                comfy.sample.sample(
                    model, noise, args.steps, args.cfg, args.sampler, args.scheduler,
                    positive, negative, latent,
                    denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, callback=None,
                    disable_pbar=True, seed=seed)
                torch.cuda.synchronize()
                runs += 1
    finally:
        for handle in handles:
            handle.remove()

    elapsed = time.perf_counter() - started
    missing = sorted(set(targets) - set(stats))
    if missing:
        # A hooked layer that never fired is a layer the converter must not calibrate from. Say
        # so loudly: it usually means the profile matched something outside the sampled path.
        print(f"warning: {len(missing)} hooked layer(s) never ran, e.g. {missing[:3]}")

    # Keyed by checkpoint name, not module name -- see MODULE_TO_FILE. Checked rather than
    # trusted: a translation that collides would silently drop layers from the calibration.
    payload = {to_file_name(args.profile, name): entry.finish() for name, entry in stats.items()}
    if len(payload) != len(stats):
        raise SystemExit(f"the module->file translation collapsed {len(stats)} layers into "
                         f"{len(payload)}; MODULE_TO_FILE for profile {args.profile!r} is wrong")
    # Quantas camadas sairam com TODA linha rotulada. Se o wrapper nunca disparasse -- outro
    # caminho de sampling, uma versao do ComfyUI que nao chama `model_function_wrapper` -- a
    # coluna sairia toda NaN e o arquivo pareceria normal. Isto e o que impede isso de passar.
    tagged = sum(1 for e in payload.values() if e.get("sigma_tagged"))
    sigmas_vistos = torch.cat([e["sample_sigma"] for e in payload.values()]) if payload else torch.zeros(0)
    finitos = sigmas_vistos[torch.isfinite(sigmas_vistos)] if sigmas_vistos.numel() else sigmas_vistos
    meta = {
        "source": str(path),
        "source_identity_sha256": source_digest,
        "sample_dtype": str(torch.bfloat16),
        "sigma_tagged_layers": tagged,
        "sigma_min": float(finitos.min()) if finitos.numel() else None,
        "sigma_max": float(finitos.max()) if finitos.numel() else None,
        "sigma_distinct": int(torch.unique(finitos).numel()) if finitos.numel() else 0,
        "profile": args.profile,
        "prompts": len(prompts),
        "seeds": args.seeds,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
        "size": args.size,
        "rows": args.rows,
        "runs": runs,
        "layers": len(payload),
        "never_ran": missing,
        "seconds": round(elapsed, 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    torch.save({"meta": meta, "layers": payload}, partial)
    partial.replace(args.out)

    if tagged != len(payload):
        print(f"warning: {len(payload) - tagged} de {len(payload)} camadas ficaram com alguma "
              "linha SEM sigma. `quant_mixed --sigma-weight` vai recusar essas camadas em vez "
              "de tratar NaN como zero.")

    print(f"\n{json.dumps(meta, indent=2)}")
    worst = sorted(payload.items(), key=lambda kv: -kv[1]["crest_p99"])[:10]
    print(f"\n{'layer':<44}{'rows':>10}{'crest p50':>12}{'crest p99':>12}{'crest max':>12}")
    for name, entry in worst:
        print(f"{name:<44}{entry['rows']:>10}{entry['crest_p50']:>12.2f}"
              f"{entry['crest_p99']:>12.2f}{entry['crest_max']:>12.2f}")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB) in {elapsed:.1f}s")
    print("crest factor is a diagnostic. The promotion decision is measured against the real "
          "kernels in tools/quant_mixed.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
