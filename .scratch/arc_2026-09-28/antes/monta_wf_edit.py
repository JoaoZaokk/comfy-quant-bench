"""Grafos API dos workflows de edição (sobem como userdata temporário para o frontend converter em UI)."""
import json, urllib.parse, urllib.request
M = "/mnt/comfy-models"
PROMPT = "Put a red knitted beanie on the man in <image1>; keep his face, pose, clothing and background unchanged."
V01, V021, T8 = ("Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors",
                 "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", "turbo8_lora_step2500.safetensors")


def n(cls, title, **inputs):
    return {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}


def comum(prefixo):
    return {
        "20": n("LoadImage", "Imagem 1 (a que vai ser editada)", image="edit_pescador.png"),
        "21": n("LoadImage", "Imagem 2 (opcional, reative com Ctrl+M)", image="edit_frasco.png"),
        "1": n("ZenImage21AdapterLoader", "Encoder zen (Qwen3.5-0.8B)", model_folder="", adapter_file=f"{M}/text_encoders/zen/adapter_v12.safetensors",
               text_encoder=f"{M}/text_encoders/qwen3.5_0.8b", dtype="bf16"),
        "2": n("ZenImage21TextEncode", "Instrução (inglês, cite <image1>/<image2>)", adapter=["1", 0], vae=["4", 0], prompt=PROMPT, negative_prompt="",
               resolution=768, width=0, height=0, image_1=["20", 0], image_2=["21", 0]),
        "3": n("UnetLoaderGGUF", "DiT Q4_1 (GGUF)", unet_name="qwen_image_2.1_bf16_Q4_1.gguf"),
        "4": n("VAELoader", "VAE", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "7": n("VAEDecode", "VAE Decode", samples=["6", 0], vae=["4", 0]),
        "8": n("SaveImage", "Salvar", images=["7", 0], filename_prefix=prefixo),
    }


def base():
    g = comum("qwen21_edicao")
    g["6"] = n("KSampler", "KSampler (25 passos)", model=["3", 0], seed=42, steps=25, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=["2", 0], negative=["2", 1], latent_image=["2", 2], denoise=1.0)
    return g


def viggle(lora, sigmas, prefixo):
    g = comum(prefixo)
    g.update({
        "10": n("ViggleTurboLora", "LoRA turbo (sem mesclar)", model=["3", 0], lora_name=lora, strength=1.0),
        "11": n("BasicGuider", "Guider (sem CFG)", model=["10", 0], conditioning=["2", 0]),
        "12": n("KSamplerSelect", "Sampler", sampler_name="euler"),
        "13": n("ViggleTurboSigmas", "Passos (sigmas)", latent=["2", 2], nodes=sigmas),
        "14": n("RandomNoise", "Seed", noise_seed=42),
        "6": n("SamplerCustomAdvanced", "Amostragem", noise=["14", 0], guider=["11", 0], sampler=["12", 0], sigmas=["13", 0], latent_image=["2", 2]),
    })
    return g


def turbo8():
    g = comum("qwen21_edicao_turbo8")
    g.update({
        "10": n("LoraLoaderModelOnly", "LoRA Turbo8", model=["3", 0], lora_name=T8, strength_model=1.0),
        "11": n("ModelSamplingFlux", "Agenda do Turbo8", model=["10", 0], max_shift=0.6935, base_shift=0.5, width=768, height=768),
        "6": n("KSampler", "KSampler (8 passos)", model=["11", 0], seed=42, steps=8, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=["2", 0], negative=["2", 1], latent_image=["2", 2], denoise=1.0),
    })
    return g


COMUM = ("### Como usar\n* **Imagem 1** é a que vai ser editada; a saída segue o formato dela, com ~768² de área.\n"
         "* **Imagem 2** (referência extra) vem desativada: reative com Ctrl+M e cite no prompt como `<image2>`.\n"
         "* Escreva a instrução em inglês citando `<image1>`, `<image2>` (\"the first image\" não funciona).\n\n"
         "### Por que 768\nNesta máquina (DiT Q4_1 + encoder zen), referências em 1024 saem com aspecto \"HDR queimado\" e "
         "ignoram a instrução; em 768 a edição sai certa. Não suba o `resolution` do nó de instrução.\n\n"
         "Testado na Arc: 1 e 2 referências. Não rode junto com o llama.cpp.")
WF = {
    "Qwen-Image 2.1 edição (25 passos)": (base(), "## Edição — 25 passos\n\n~95 s por edição (1 referência).\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 4 passos (Viggle v0.1)": (viggle(V01, "1.0, 0.75, 0.5, 0.25", "qwen21_edicao_turbo4"),
        "## Edição turbo — 4 passos (Viggle v0.1)\n\n~50 s por edição.\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 6 passos (Viggle v0.2.1)": (viggle(V021, "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25", "qwen21_edicao_turbo6"),
        "## Edição turbo — 6 passos (Viggle v0.2.1) — recomendado\n\n~55-80 s por edição.\n\n" + COMUM),
    "Qwen-Image 2.1 edição turbo 8 passos (Turbo8)": (turbo8(),
        "## Edição turbo — 8 passos (Turbo8)\n\n~60-95 s por edição. A agenda `ModelSamplingFlux` usa 768×768; "
        "se a imagem 1 não for quadrada, ajuste largura/altura dela para o tamanho da saída.\n\n" + COMUM),
}
for nome, (g, nota) in WF.items():
    body = json.dumps({"api": g, "nota": nota}, ensure_ascii=False).encode()
    url = "http://127.0.0.1:8188/api/userdata/" + urllib.parse.quote("tmp_api/" + nome + ".json", safe="") + "?overwrite=true"
    urllib.request.urlopen(urllib.request.Request(url, data=body, method="POST"))
    print("ok", nome)
