"""O braco de comparacao do 10Eros v1.5 W4A8, sem o confounder de versao.

Eu tinha comecado a baixar o `10Eros_v1.4_DMD_int8_convrot` que mora no repo do autor, como braco
barato no lugar do BF16 de 42,97 GiB (que nao abre aqui pelo leitor normal: 2x em commit). Errado:
**aquele arquivo e da v1.4 mesclada com DMD**, e o nosso W4A8 e da **v1.5**. Comparar os dois mede
versao E formato ao mesmo tempo, que e exatamente o erro de "A/B so vale se os dois tomaram o mesmo
caminho" -- com o eixo segurado sendo a base do modelo.

O arquivo certo existe: `CornLogic/10EROS-INT8 -> 10Eros_v1.5_INT8_TFO.safetensors`, INT8 da MESMA
base v1.5, **sem** merge de DMD, 25,03 GB. Mesmos pesos de partida que o nosso, formato diferente.

Junto vem duas coisas baratas e informativas:
  o manifest de quantizacao do fp8 do LokkenJP (6,2 MB de JSON por camada) -- um terceiro fazendo o
    mesmo trabalho neste mesmo modelo, com o mapa dele aberto
  o workflow que o LokkenJP adaptou para a v1.5, para comparar com o V5 oficial
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"F:\hf-cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import hf_hub_download

ALVOS = [
    ("LokkenJP/10EROS_1.5_fp8_exp_learned",
     "10Eros_v1.5_fp8mixed_experimental_learned.safetensors.quantization.json",
     6_242_455, Path(r"F:\COMFY_PORTABLE\bench\10eros"), "manifest de quant do fp8 de terceiro"),
    ("LokkenJP/10EROS_1.5_fp8_exp_learned",
     "10Eros_v1.5_fp8mixed_experimental_learned.safetensors.sha256",
     120, Path(r"F:\COMFY_PORTABLE\bench\10eros"), "sha256 declarado do fp8"),
    ("LokkenJP/10EROS_1.5_fp8_exp_learned", "README.md",
     11_242, Path(r"F:\COMFY_PORTABLE\bench\10eros\lokkenjp"), "README do fp8 de terceiro"),
    ("LokkenJP/10EROS_1.5_fp8_exp_learned",
     "workflows/10Eros_10SNodes_I2V_Basic_DMD_V5_1.5fp8.json",
     84_551, Path(r"F:\COMFY_PORTABLE\bench\10eros"), "workflow adaptado para a v1.5"),
    ("CornLogic/10EROS-INT8", "10Eros_v1.5_INT8_TFO.safetensors",
     25_032_524_432, Path(r"P:\ComfyBench\diffusion_models"), "INT8 da MESMA base v1.5 -- o braco"),
]

for repo, arq, esperado, dest, rotulo in ALVOS:
    dest.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo_id=repo, filename=arq, local_dir=str(dest))
    n = Path(p).stat().st_size
    print(f"{rotulo:42s} {n:>15,}  esperado {esperado:>15,}  "
          f"{'OK' if n == esperado else 'DIVERGE'}", flush=True)
    print(f"{'':42s} {p}", flush=True)
print("BRACO_V15_FIM")
