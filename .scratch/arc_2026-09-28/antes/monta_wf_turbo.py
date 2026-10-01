"""Gera os grafos API dos workflows turbo e sobe como userdata temporário para o frontend converter em UI."""
import json, urllib.parse, urllib.request
M = "/mnt/comfy-models"
PROMPT = 'A minimalist poster with the headline "SLOW MORNINGS" in bold serif letters above a small line drawing of a coffee cup, cream background'
V01, V021, T8 = ("Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
                 "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", "turbo8_lora_step2500.safetensors")


def n(cls, title, **inputs):
    return {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}


def comum():
    return {
        "1": n("ZenImage21AdapterLoader", "Encoder zen (Qwen3.5-0.8B)", model_folder="", adapter_file=f"{M}/text_encoders/zen/adapter_v12.safetensors",
               text_encoder=f"{M}/text_encoders/qwen3.5_0.8b", dtype="bf16"),
        "2": n("ZenImage21TextEncode", "Prompt (inglês)", adapter=["1", 0], prompt=PROMPT, negative_prompt="", resolution=1024, width=1024, height=1024),
        "3": n("UnetLoaderGGUF", "DiT Q4_1 (GGUF)", unet_name="qwen_image_2.1_bf16_Q4_1.gguf"),
        "4": n("VAELoader", "VAE", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "7": n("VAEDecode", "VAE Decode", samples=["6", 0], vae=["4", 0]),
    }


def viggle(lora, sigmas, prefixo):
    g = comum()
    g.update({
        "10": n("ViggleTurboLora", "LoRA turbo (sem mesclar)", model=["3", 0], lora_name=lora, strength=1.0),
        "11": n("BasicGuider", "Guider (sem CFG)", model=["10", 0], conditioning=["2", 0]),
        "12": n("KSamplerSelect", "Sampler", sampler_name="euler"),
        "13": n("ViggleTurboSigmas", "Passos (sigmas)", latent=["2", 2], nodes=sigmas),
        "14": n("RandomNoise", "Seed", noise_seed=42),
        "6": n("SamplerCustomAdvanced", "Amostragem", noise=["14", 0], guider=["11", 0], sampler=["12", 0], sigmas=["13", 0], latent_image=["2", 2]),
        "8": n("SaveImage", "Salvar", images=["7", 0], filename_prefix=prefixo),
    })
    return g


def turbo8():
    g = comum()
    g.update({
        "10": n("LoraLoaderModelOnly", "LoRA Turbo8", model=["3", 0], lora_name=T8, strength_model=1.0),
        "11": n("ModelSamplingFlux", "Agenda do Turbo8", model=["10", 0], max_shift=0.6935, base_shift=0.5, width=1024, height=1024),
        "6": n("KSampler", "KSampler (8 passos)", model=["11", 0], seed=42, steps=8, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=["2", 0], negative=["2", 1], latent_image=["2", 2], denoise=1.0),
        "8": n("SaveImage", "Salvar", images=["7", 0], filename_prefix="qwen21_turbo8"),
    })
    return g


COMUM = ("Encoder zen + DiT Q4_1 GGUF + VAE, 1024², cfg 1, sem prompt negativo. Prompt em inglês; "
         "números em letreiros podem sair errados. Não rode junto com o llama.cpp.")
WF = {
    "Qwen-Image 2.1 turbo 4 passos (Viggle v0.1)": (viggle(V01, "1.0, 0.75, 0.5, 0.25", "qwen21_turbo4"),
        "## Turbo 4 passos — Viggle v0.1 (r64)\n\n~30 s por imagem na Arc (21 s de amostragem).\n\n"
        "LoRA de 4 passos da Viggle. Mais rápido, um pouco mais estilizado que o modelo base. "
        "Não troque pela LoRA v0.2.1 com 4 passos: o texto sai embaralhado.\n\n" + COMUM),
    "Qwen-Image 2.1 turbo 6 passos (Viggle v0.2.1)": (viggle(V021, "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25", "qwen21_turbo6"),
        "## Turbo 6 passos — Viggle v0.2.1 (r128) — recomendado\n\n~40 s por imagem na Arc (31 s de amostragem).\n\n"
        "Para 8 passos (texto miúdo mais limpo, ~48 s), troque os sigmas por:\n`1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125`\n\n"
        "Regra do autor: só mexa na ponta de ruído alto; mantenha `0.875, 0.75, 0.5, 0.25`.\n\n" + COMUM),
    "Qwen-Image 2.1 turbo 8 passos (Turbo8)": (turbo8(),
        "## Turbo 8 passos — Turbo8 (chriswritescode)\n\n~46 s por imagem na Arc (44 s de amostragem).\n\n"
        "LoRA comum (LoraLoaderModelOnly) com a agenda `ModelSamplingFlux` 0.6935/0.5; mantenha largura/altura "
        "iguais às do nó de prompt. Mantenha 8 passos, cfg 1, euler/simple.\n\n" + COMUM),
}
for nome, (g, nota) in WF.items():
    body = json.dumps({"api": g, "nota": nota}, ensure_ascii=False).encode()
    url = "http://127.0.0.1:8188/api/userdata/" + urllib.parse.quote("tmp_api/" + nome + ".json", safe="") + "?overwrite=true"
    urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"))
    print("ok", nome)
