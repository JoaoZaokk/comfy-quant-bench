

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
