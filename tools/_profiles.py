"""Perfis de arquitetura: QUAIS camadas de um checkpoint cada ferramenta quantiza. Um lugar so.

Ate 2026-09-29 esta tabela existia em QUATRO copias que ja divergiam (revisao de 2026-09-29,
achado 6): `quant_w4a4.PROFILE_PATTERNS` (com `EXCLUSIONS`), `quant_w4a8.PROFILE_PATTERNS`,
`calibrate_activations.PROFILE_PATTERNS` (nomes de MODULO) e `PROFILE_FILE_PATTERNS`. As regex de
LTX, Qwen-Image-2.1 e HunyuanVideo eram copiadas literalmente de um arquivo para o outro ("the same
regex quant_w4a8.py uses"); `detect_profile` tambem, e a copia do w4a4 nao tinha o ramo estrutural
de LTX -- `quant_w4a4 --profile auto` numa LTX levantava ValueError enquanto o w4a8 detectava; e a
segunda rede (`EXCLUSIONS`) so existia no w4a4.

Tres vocabularios, porque sao tres perguntas diferentes, agora DERIVADOS um do outro em vez de
copiados:

    MODULE_PATTERNS   nome de MODULO no modelo carregado -- o que `calibrate_activations` engancha.
    FILE_PATTERNS     nome da CHAVE no checkpoint, sem o `.weight` -- o que `quant_mixed` casa. Igual
                      a MODULE_PATTERNS exceto onde o ComfyUI renomeia ao carregar (MODULE_TO_FILE).
    WEIGHT_PATTERNS   FILE_PATTERNS + `\\.weight$`, mais os text encoders (que so existem do lado do
                      conversor) -- o que w4a4/w4a8/int8/awq/gguf casam, tensor a tensor.

`PROFILE_PATTERNS` (o nome antigo, importado por weight_balance, quant_gguf...) continua sendo o
subconjunto de WEIGHT_PATTERNS que os conversores de formato unico oferecem em `--profile`, com as
MESMAS seis chaves de antes. Conferido na migracao: as regex derivadas sao, caractere a caractere,
as que as copias antigas carregavam, e a selecao nos headers reais do disco nao mudou (ver
`.scratch/revisao_2026-09-29/resultado_conversao.md`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

HIGH_PRECISION_DTYPES = frozenset({"BF16", "F16", "F32"})

# ---------------------------------------------------------------- nomes de MODULO (calibracao)

# Kept identical to the converter's, and imported by it, so a layer can never be calibrated under
# one definition and quantized under another.
MODULE_PATTERNS = {
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
    # Qwen-Image-2.1 (`comfy/ldm/qwen_image21/model.py`), 32 blocos de corrente unica x 6 Linears
    # = 192. Modulo e arquivo tem o mesmo nome. Derivado do `qwen_image_2.1_int8_convrot` da
    # Comfy-Org (192 `comfy_quant`, sem falso positivo nem negativo) e do mixed da NidAll (as
    # mesmas 192). `gate_up` e o layout fundido da Comfy-Org; `proj`/`gate_layer`, o do diffusers.
    # FORA: `img_in`, `txt_in.*`, `modulation.1` (global, nao por bloco), o embedder de timestep,
    # `norm_out.linear` e `proj_out`.
    "qwen_image21": re.compile(
        r"^transformer_blocks\.\d+\.(?:attn\.(?:to_[qkv]|to_out\.0)|img_mlp\.(?:gate_up|proj|gate_layer|out))$"
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
# from their module naming need an entry; the rest fall back to MODULE_PATTERNS.
FILE_PATTERNS = dict(MODULE_PATTERNS)
FILE_PATTERNS["hunyuan_video_15"] = re.compile(
    r"^double_blocks\.\d+\.(?:(?:img|txt)_attn_(?:qkv|proj)|(?:img|txt)_mlp\.fc[12])$"
)
FILE_PATTERNS["wan_2_1"] = re.compile(
    r"^(?:model\.diffusion_model\.)?blocks\.\d+\.(?:(?:self|cross)_attn\.[qkvo]|ffn\.[02])$"
)
FILE_PATTERNS["wan_2_2"] = re.compile(
    r"^(?:model\.diffusion_model\.)?blocks\.\d+\."
    r"(?:(?:self|cross)_attn\.(?:[qkvo]|[kv]_img)|ffn\.[02])$"
)


# ---------------------------------------------------------------- nomes de TENSOR (conversores)

# Os tres text encoders so aparecem do lado do conversor: `calibrate_activations` calibra modelos
# de difusao. `gemma` e `qwen` sao a MESMA regex sob dois nomes de proposito -- o nome escolhe o
# rotulo no sidecar e a lista de exclusoes, e a autodeteccao decide pelo nome do arquivo.
_DECODER = (r"\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
            r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$")
TEXT_ENCODER_PATTERNS = {
    "gemma": re.compile(r"^model\.layers" + _DECODER),
    "qwen": re.compile(r"^model\.layers" + _DECODER),
    # Qwen3-VL as a text encoder -- Krea 2's conditioner. Same decoder as `qwen`, one segment
    # deeper: `model.language_model.layers.N` because the checkpoint carries a vision tower beside
    # the language model. That extra segment is exactly why the `qwen` profile matches ZERO layers
    # here; checked against the header of qwen3vl_4b_bf16 before this entry was written, not
    # assumed from the family name.
    "qwen3vl": re.compile(r"^model\.language_model\.layers" + _DECODER),
}


def _weight_pattern(stem: re.Pattern) -> re.Pattern:
    if not stem.pattern.endswith("$"):
        raise ValueError(f"file pattern without a final '$': {stem.pattern!r}")
    return re.compile(stem.pattern[:-1] + r"\.weight$")


WEIGHT_PATTERNS: dict[str, re.Pattern] = {
    **{name: _weight_pattern(pattern) for name, pattern in FILE_PATTERNS.items()},
    **TEXT_ENCODER_PATTERNS,
}

# Os perfis que os conversores de formato unico (quant_w4a4, quant_w4a8, quant_int8,
# quant_awq_w4a16, quant_gguf) oferecem. Mesmas seis chaves de antes da consolidacao; abrir outro
# perfil para eles e uma decisao de receita, nao efeito colateral de mover a tabela.
#
# ltx_2_5: derived from Lightricks' own ltx-2.5-...-comfy-int8-convrot checkpoint rather than
# guessed: this pattern selects exactly the 1440 Linears they quantized, with no false positives
# and no false negatives against their shipped `comfy_quant` markers. What they leave out is as
# informative as what they include -- 304 `to_gate_logits`, every adaLN/timestep embedder, and the
# patchify/proj_out pair stay in full precision. Note they ship int8 weights, not int4: converting
# the same layers to W4A8/W4A4 is strictly more aggressive than what Lightricks considered safe,
# so A/B it before trusting the output. Verified 2026-09-13 on both headers: 2.3 and 2.5 share 76
# identical 2-D weight families and this selects 1440/1772 on each, on the transformer-only file
# (2.5) and on the single-file checkpoint (2.3, `model.diffusion_model.` prefix) alike.
#
# hunyuan_video_15: HunyuanVideo 1.5 double-stream blocks, in the CHECKPOINT's own naming;
# HunyuanVideo.process_unet_state_dict rewrites them to the ComfyUI module names after
# convert_old_quants has injected the .comfy_quant keys, and its substring replacements
# ("_attn_qkv." -> "_attn.qkv.", "mlp.fc1." -> "mlp.0.", ...) carry the injected .comfy_quant and
# .weight_scale keys along with the weights.
#
# qwen_image21: Qwen-Image-2.1 single-stream DiT. Derived from Comfy-Org's own int8-convrot and
# NidAll's mixed checkpoints (26/09): both quantize exactly these 6 Linears per block (192), and
# keep img_in, txt_in, modulation, the timestep embedder, norm_out and proj_out in BF16.
CONVERTER_PROFILES = ("ltx_2_5", "hunyuan_video_15", "qwen_image21", "gemma", "qwen", "qwen3vl")
PROFILE_PATTERNS: dict[str, re.Pattern] = {name: WEIGHT_PATTERNS[name] for name in CONVERTER_PROFILES}

# Segunda rede, INDEPENDENTE da allowlist: uma camada que casa a regex mas contem um destes trechos
# nao e selecionada. Hoje nenhuma allowlist deixa passar nada daqui (conferido na migracao contra os
# headers reais do disco: selecao identica com a rede ligada em todos os conversores); ela existe
# para o dia em que alguem escrever uma regex mais frouxa. Ate 2026-09-29 so o quant_w4a4 a aplicava.
EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "qwen_image21": ("norm", "modulation", "img_in", "txt_in", "time_text_embed", "proj_out"),
    "gemma": ("embed_tokens", "norm", "lm_head", "vision"),
    "qwen": ("embed_tokens", "norm", "lm_head", "visual", "vision"),
    "qwen3vl": ("embed_tokens", "norm", "lm_head", "visual", "vision"),
    # adaLN modulation drives every block's conditioning, so *_mod.linear stays high precision
    # along with the norms, the embedders, the token refiner, and the byt5/vision/time adapters.
    "hunyuan_video_15": ("_mod.linear", "_norm", "norm", "img_in", "txt_in", "byt5_in", "vision_in",
                         "time_in", "final_layer", "_embedding", "task_bias"),
    # Modulation, norms, patch/caption projections, the VAEs and the vocoder of the single-file
    # checkpoint. The allowlist already excludes all of them.
    "ltx_2_5": ("norm", "adaln_single", "scale_shift", "proj_in", "proj_out", "caption_projection",
                "patchify", "vae.", "audio_vae.", "vocoder.", "text_embedding_projection",
                "to_gate_logits"),
}


def is_qwen_image21(name_set: set[str]) -> bool:
    """The same keys `comfy.model_detection` uses for image_model == "qwen_image21"."""
    return ({"txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.attn.norm_q.weight",
             "img_in.weight", "proj_out.weight"} <= name_set
            and bool({"transformer_blocks.0.img_mlp.gate_up.weight",
                      "transformer_blocks.0.img_mlp.proj.weight"} & name_set))


def detect_profile(path: Path, names: list[str]) -> str:
    """Perfil a partir do checkpoint: estrutura primeiro, nome do arquivo so para os text encoders.

    Era a copia do quant_w4a8; a do quant_w4a4 nao tinha o ramo de LTX. Decisao do dono em
    2026-09-29: a autodeteccao de LTX no w4a4 passa a funcionar.
    """
    name_set = set(names)
    # LTX-2.5 is the only architecture here with paired audio<->video cross-attention next to a
    # separate embeddings connector stack, so those two keys identify it without the file name.
    for prefix in ("", "model.diffusion_model."):
        if {f"{prefix}transformer_blocks.0.audio_to_video_attn.to_q.weight",
                f"{prefix}video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"} <= name_set:
            return "ltx_2_5"
    # Mirrors the HunyuanVideo branch of comfy.model_detection, so the profile is derived from the
    # checkpoint rather than from its file name.
    if {"txt_in.individual_token_refiner.blocks.0.norm1.weight",
            "double_blocks.0.img_attn_qkv.weight"} <= name_set:
        return "hunyuan_video_15"
    if is_qwen_image21(name_set):
        return "qwen_image21"
    lowered = Path(path).name.lower()
    # `qwen3vl` before `qwen`: the VL checkpoint has no `model.layers.` at all, so the order only
    # matters if a future file carries both, and then the deeper name is the right answer.
    if "qwen" in lowered and any(n.startswith("model.language_model.layers.") for n in names):
        return "qwen3vl"
    for profile in ("gemma", "qwen"):
        if profile in lowered and any(n.startswith("model.layers.") for n in names):
            return profile
    raise ValueError("auto-detection found no supported profile; pass --profile explicitly "
                     "after verifying the architecture")


def select_layers(header: dict, profile: str,
                  accepts: Callable[[list[int]], bool] | None = None) -> list[str]:
    """Os pesos 2-D de alta precisao que o perfil seleciona, na ordem do header.

    Allowlist (`WEIGHT_PATTERNS[profile]`) primeiro, depois `EXCLUSIONS` como segunda rede
    independente. `accepts(shape)` e a restricao do FORMATO (divisibilidade de K), que mora em
    `_formats` -- o perfil diz quais camadas, o formato diz quais formas ele consegue guardar.
    """
    pattern = WEIGHT_PATTERNS[profile]
    excluded = EXCLUSIONS.get(profile, ())
    selected = []
    for name, info in header.items():
        shape = info["shape"]
        if (pattern.fullmatch(name)
                and info["dtype"] in HIGH_PRECISION_DTYPES
                and len(shape) == 2
                and (accepts is None or accepts(shape))
                and not any(token in name for token in excluded)):
            selected.append(name)
    return selected
