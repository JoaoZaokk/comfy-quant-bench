"""Sobe o W4A8 do LTX 2.5, com a licenca completa e a prova visual.

A LICENCA E CONDICAO, NAO ENFEITE
---------------------------------
A LTX-2.x Community License, Secao 3, permite redistribuir derivados -- lido no texto que vem
dentro do proprio `__metadata__` do checkpoint, nao num resumo -- **desde que**:

  3.2  o derivado seja distribuido EXCLUSIVAMENTE sob os mesmos termos, com copia COMPLETA do
       acordo junto;
  3.3  os arquivos modificados carreguem aviso destacado de que foram alterados;
  3.4  avisos de copyright e atribuicao sejam mantidos.

Por isso o `LICENSE_LTX_2x_COMMUNITY.txt` (35.143 B) sobe junto e o README carrega a declaracao
de mudanca. Subir o peso sem eles violaria a licenca do modelo de origem.

O QUE SOBE
----------
O peso W4A8, o sidecar, o README em ingles, a licenca, e a folha de contato dos TRES bracos --
BF16, int8 da Lightricks e o nosso -- porque a afirmacao central deste card e que o nosso e o
mais distante do original, e isso o leitor tem de poder conferir com os olhos.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import HfApi  # noqa: E402

REPO = "JoaoZaokk/LTX-2.5-22B-distilled-W4A8-ConvRot"
RAIZ = r"F:\COMFY_PORTABLE"
CARD = os.path.join(RAIZ, "bench", "hf", "ltx25-22b-w4a8")
FONTE = r"D:\ComfyUI-Models\diffusion_models"

ARQUIVOS = [
    (os.path.join(FONTE, "ltx-2.5-22b-distilled-transformer-bf16_w4a8.safetensors"),
     "ltx-2.5-22b-distilled-transformer-w4a8.safetensors"),
    (os.path.join(FONTE, "ltx-2.5-22b-distilled-transformer-bf16_w4a8.quant.json"),
     "ltx-2.5-22b-distilled-transformer-w4a8.quant.json"),
    (os.path.join(CARD, "LICENSE_LTX_2x_COMMUNITY.txt"), "LICENSE_LTX_2x_COMMUNITY.txt"),
    (os.path.join(CARD, "comparacao_10s_tres_bracos.png"), "comparacao_10s_tres_bracos.png"),
    (os.path.join(CARD, "README.md"), "README.md"),
]

api = HfApi()
print(f"autenticado como {api.whoami().get('name')}", flush=True)
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
print(f"repo: https://huggingface.co/{REPO}", flush=True)

for local, remoto in ARQUIVOS:
    if not os.path.exists(local):
        print(f"FALTA {local}", flush=True)
        sys.exit(1)
    n = os.path.getsize(local)
    print(f"subindo {remoto} ({n:,} B) ...", flush=True)
    t0 = time.time()
    api.upload_file(path_or_fileobj=local, path_in_repo=remoto, repo_id=REPO, repo_type="model")
    print(f"  ok em {time.time() - t0:.0f}s", flush=True)

print("LTX25_UPLOAD_OK", flush=True)
