"""Loader do Qwen-Image-2.1 SVDQuant INT4 (formato `qwen21-nunchaku-svdq-int4-v1`, ex. mesmertech/Mesmer-Image-21-Nunchaku).

O arquivo guarda o transformer do diffusers com as 224 lineares dos blocos (attn.to_q/k/v/to_out.0 e
img_mlp.proj/out/gate_layer) empacotadas para o `SVDQW4A4Linear` do Nunchaku (int4, rank no manifesto); o resto e
BF16 comum. O runtime publicado pelo autor e uma imagem Docker; aqui o mesmo contrato roda dentro do modelo NATIVO
do ComfyUI (`comfy.ldm.qwen_image21`), que usa os mesmos nomes de camada: o ComfyUI monta o modelo, as 224 lineares
sao trocadas pelas do Nunchaku e o arquivo e carregado com strict. Tudo o mais (atencao, cache de prefixo, VAE,
text encoder, sampler) e o caminho oficial do ComfyUI.

As lineares do Nunchaku nao aceitam o carregamento parcial/offload por camada do ComfyUI: o patcher move o modelo
inteiro para a placa (como o ComfyUI-nunchaku faz). LoRA nao e suportado.
"""
import json
import logging
import re
import struct

import torch

import comfy.model_detection
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths

logger = logging.getLogger("comfy-qwen21-nunchaku")

FORMATO = "qwen21-nunchaku-svdq-int4-v1"
LINEAR = re.compile(r"^transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v|to_out\.0)|img_mlp\.(?:proj|out|gate_layer))$")


def ler_manifesto(caminho):
    with open(caminho, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get("__metadata__") or {}
    man = json.loads(meta.get("manifest", "{}"))
    if man.get("backend_format") != FORMATO or man.get("complete") is not True:
        raise ValueError(f"{caminho}: nao e um checkpoint {FORMATO} completo (backend_format={man.get('backend_format')})")
    camadas = man.get("layers", {})
    ruins = [k for k in camadas if not LINEAR.fullmatch(k)]
    if len(camadas) != 224 or ruins:
        raise ValueError(f"manifesto com {len(camadas)} camadas (esperado 224); fora do padrao: {ruins[:3]}")
    for k, i in camadas.items():
        if i.get("precision", "int4") != "int4":
            raise ValueError(f"{k}: precisao {i.get('precision')} (so int4 roda em Ampere/Ada; o fp4 e so Blackwell)")
    return camadas


class Qwen21NunchakuPatcher(comfy.model_patcher.ModelPatcher):
    """Move o modelo inteiro; sem offload por camada (as lineares do Nunchaku nao tem os hooks de cast do ComfyUI)."""

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        self.model.diffusion_model.to(device_to)
        self.model.device = device_to
        self.model.model_loaded_weight_memory = self.model_size()
        self.model.model_lowvram = False
        self.model.lowvram_patch_counter = 0

    def detach(self, unpatch_all=True):
        self.model.diffusion_model.to(self.offload_device)
        self.model.device = self.offload_device
        self.model.model_loaded_weight_memory = 0
        return self.model

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        if memory_to_free <= 0:
            return 0
        self.detach()
        return self.model_size()

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        self.load(device_to)
        return self.model_size()

    # `load` acima nao aplica patches de peso: uma LoRA seria aceita (nas camadas BF16 que casam) e ignorada
    # sem aviso. Recusar e' o comportamento honesto enquanto LoRA nao for suportado.
    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        if patches:
            raise RuntimeError("Qwen-Image 2.1 Nunchaku INT4: LoRA/patches de peso nao sao suportados "
                               "(o carregamento inteiro para a placa nao os aplica)")
        return []

    def add_hook_patches(self, hook, patches, strength_patch=1.0, strength_model=1.0):
        if patches:
            raise RuntimeError("Qwen-Image 2.1 Nunchaku INT4: patches de peso por hook nao sao suportados")
        return []

    def add_weight_wrapper(self, name, function):
        raise RuntimeError("Qwen-Image 2.1 Nunchaku INT4: wrappers de peso nao sao suportados")


class Qwen21NunchakuLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("diffusion_models"),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "Qwen-Image-2.1 SVDQuant INT4 (qwen21-nunchaku-svdq-int4-v1) no modelo nativo do ComfyUI, com kernels do Nunchaku."

    def load(self, unet_name):
        from nunchaku.models.linear import SVDQW4A4Linear

        caminho = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        camadas = ler_manifesto(caminho)
        sd = comfy.utils.load_torch_file(caminho)
        # a deteccao do ComfyUI olha `<linear>.weight`; as lineares empacotadas nao tem, entao entram formas em meta
        deteccao = dict(sd)
        for k, i in camadas.items():
            deteccao[k + ".weight"] = torch.empty(i["out_features"], i["in_features"], device="meta")
        config = comfy.model_detection.model_config_from_unet(deteccao, "")
        if config is None or config.unet_config.get("image_model") != "qwen_image21":
            raise RuntimeError("o ComfyUI nao reconheceu o arquivo como Qwen-Image-2.1")
        if config.unet_config.get("fused_mlp"):
            raise RuntimeError("config com MLP fundida; o checkpoint INT4 tem proj/gate_layer separados")

        load_device, offload_device = mm.get_torch_device(), mm.unet_offload_device()
        config.set_inference_dtype(torch.bfloat16, None)
        # montado em meta: os pesos BF16 das 224 lineares seriam alocados so para serem trocados (~13 GB)
        model = config.get_model({}, "", device=torch.device("meta"))
        dm = model.diffusion_model
        for k, i in camadas.items():
            pai, filho = k.rsplit(".", 1)
            mod_pai = dm.get_submodule(pai)
            orig = mod_pai.get_submodule(filho)
            if (orig.in_features, orig.out_features) != (i["in_features"], i["out_features"]):
                raise RuntimeError(f"{k}: modelo {orig.in_features}x{orig.out_features} != checkpoint {i['in_features']}x{i['out_features']}")
            mod_pai._modules[filho] = SVDQW4A4Linear(i["in_features"], i["out_features"], rank=i["rank"], bias=i["bias"],
                                                     precision="int4", act_unsigned=False, torch_dtype=torch.bfloat16,
                                                     device="meta")
        esperado = set(dm.state_dict())
        faltam, sobram = esperado - set(sd), set(sd) - esperado
        if faltam or sobram:
            raise RuntimeError(f"checkpoint x modelo: faltam {sorted(faltam)[:5]} ({len(faltam)}), sobram {sorted(sobram)[:5]} ({len(sobram)})")
        dm.load_state_dict(sd, strict=True, assign=True)
        del sd, deteccao
        ficou = [n for n, t in list(dm.named_parameters()) + list(dm.named_buffers()) if t.is_meta]
        if ficou:
            raise RuntimeError(f"tensores sem valor depois do carregamento: {ficou[:5]} ({len(ficou)})")
        model.to(offload_device)
        for k in camadas:
            m = dm.get_submodule(k)
            if m.qweight.dtype != torch.int8 or m.wscales.dtype != torch.bfloat16:
                raise RuntimeError(f"{k}: dtypes empacotados errados ({m.qweight.dtype}, {m.wscales.dtype})")
        dm.eval().requires_grad_(False)
        logger.info(f"Qwen-Image-2.1 INT4: {len(camadas)} lineares SVDQW4A4 (rank {next(iter(camadas.values()))['rank']})")
        return (Qwen21NunchakuPatcher(model, load_device=load_device, offload_device=offload_device),)


NODE_CLASS_MAPPINGS = {"Qwen21NunchakuLoader": Qwen21NunchakuLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Qwen21NunchakuLoader": "Qwen-Image 2.1 Nunchaku INT4 Loader"}
