"""Grafos API do Qwen-Image 2.1 leve na Arc (só stdlib: roda com o python3 do sistema na VM).

Fonte única dos arquivos de modelo, das receitas de amostragem e dos montadores usados por roda_*.py e monta_wf_*.py.
"""
M = "/mnt/comfy-models"
DIT = "qwen_image_2.1_bf16_Q4_1.gguf"
VAE = "qwen_image_2.1_vae_bf16.safetensors"
V01 = "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
V021 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
T8 = "turbo8_lora_step2500.safetensors"

# Receitas de amostragem por tag. viggle: LoRA sem mesclar + sigmas próprios (BasicGuider, sem CFG).
# flux: LoRA comum + ModelSamplingFlux 0.6935/0.5 (o shift só depende da área, não do formato).
RECEITAS = {
    "base25": {"passos": 25},
    "base25s5": {"passos": 25, "shift": 5.0},
    "base4": {"passos": 4},  # sem LoRA: só para isolar o NaN (imagem ruim por construção)
    "v01_4": {"viggle": V01, "sigmas": "1.0, 0.75, 0.5, 0.25"},
    "sig4": {"viggle": None, "sigmas": "1.0, 0.75, 0.5, 0.25"},  # caminho de amostragem da Viggle sem LoRA (NaN)
    "v01_4_f0": {"viggle": V01, "sigmas": "1.0, 0.75, 0.5, 0.25", "forca": 0.0},  # hooks ativos, LoRA zerada (NaN)
    "viggle4": {"viggle": V021, "sigmas": "1.0, 0.75, 0.5, 0.25"},  # quebra texto; fica só para reproduzir
    "viggle6": {"viggle": V021, "sigmas": "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"},
    "viggle8": {"viggle": V021, "sigmas": "1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125"},
    "turbo8": {"flux": T8, "passos": 8},
}


def n(cls, title=None, **inputs):
    no = {"class_type": cls, "inputs": inputs}
    if title:
        no["_meta"] = {"title": title}
    return no


def carregadores():
    return {
        "1": n("ZenImage21AdapterLoader", "Encoder zen (Qwen3.5-0.8B)", model_folder="",
               adapter_file=f"{M}/text_encoders/zen/adapter_v12.safetensors",
               text_encoder=f"{M}/text_encoders/qwen3.5_0.8b", dtype="bf16"),
        "3": n("UnetLoaderGGUF", "DiT Q4_1 (GGUF)", unet_name=DIT),
        "4": n("VAELoader", "VAE", vae_name=VAE),
    }


def texto(prompt, lado=1024, latente_vazio=False, titulo="Prompt (inglês)"):
    """Texto para imagem. latente_vazio reproduz a bateria de 12 (EmptyLatentImage em vez do latente do zen)."""
    g = carregadores()
    g["2"] = n("ZenImage21TextEncode", titulo, adapter=["1", 0], prompt=prompt, negative_prompt="",
               resolution=lado, width=lado, height=lado)
    latente = ["2", 2]
    if latente_vazio:
        g["5"] = n("EmptyLatentImage", width=lado, height=lado, batch_size=1)
        latente = ["5", 0]
    return g, latente


def edicao(prompt, imagens, resolucao=768, titulo="Instrução (inglês, cite <image1>/<image2>)", titulos=()):
    """Edição com 1-2 referências. Em 1024 as referências saem "HDR queimado" nesta máquina; o padrão é 768."""
    g = carregadores()
    refs = {}
    for i, arquivo in enumerate(imagens):
        g[f"2{i}"] = n("LoadImage", titulos[i] if i < len(titulos) else None, image=arquivo)
        refs[f"image_{i + 1}"] = [f"2{i}", 0]
    g["2"] = n("ZenImage21TextEncode", titulo, adapter=["1", 0], vae=["4", 0], prompt=prompt, negative_prompt="",
               resolution=resolucao, width=0, height=0, **refs)
    return g, ["2", 2]


def amostra(g, latente, tag, prefixo, semente=42, area=1024):
    """Acrescenta amostragem (receita `tag`), decode e SaveImage. `area` é o lado equivalente da saída (Turbo8)."""
    r = RECEITAS[tag]
    if "viggle" in r:
        modelo = ["3", 0]
        if r["viggle"]:
            g["10"] = n("ViggleTurboLora", "LoRA turbo (sem mesclar)", model=modelo, lora_name=r["viggle"],
                        strength=r.get("forca", 1.0))
            modelo = ["10", 0]
        g.update({
            "11": n("BasicGuider", "Guider (sem CFG)", model=modelo, conditioning=["2", 0]),
            "12": n("KSamplerSelect", "Sampler", sampler_name="euler"),
            "13": n("ViggleTurboSigmas", "Passos (sigmas)", latent=latente, nodes=r["sigmas"]),
            "14": n("RandomNoise", "Seed", noise_seed=semente),
            "6": n("SamplerCustomAdvanced", "Amostragem", noise=["14", 0], guider=["11", 0], sampler=["12", 0],
                   sigmas=["13", 0], latent_image=latente),
        })
    else:
        modelo = ["3", 0]
        if "flux" in r:
            g["10"] = n("LoraLoaderModelOnly", "LoRA Turbo8", model=modelo, lora_name=r["flux"], strength_model=1.0)
            g["11"] = n("ModelSamplingFlux", "Agenda do Turbo8", model=["10", 0], max_shift=0.6935, base_shift=0.5,
                        width=area, height=area)
            modelo = ["11", 0]
        elif "shift" in r:
            g["9"] = n("ModelSamplingAuraFlow", model=modelo, shift=r["shift"])
            modelo = ["9", 0]
        g["6"] = n("KSampler", f"KSampler ({r['passos']} passos)", model=modelo, seed=semente, steps=r["passos"],
                   cfg=1.0, sampler_name="euler", scheduler="simple", positive=["2", 0], negative=["2", 1],
                   latent_image=latente, denoise=1.0)
    g["7"] = n("VAEDecode", "VAE Decode", samples=["6", 0], vae=["4", 0])
    g["8"] = n("SaveImage", "Salvar", images=["7", 0], filename_prefix=prefixo)
    return g
