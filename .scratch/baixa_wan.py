"""Baixa a suite Wan 2.2 que esta bancada consegue testar, em ordem de risco crescente.

A ESCOLHA, E O QUE FICOU DE FORA
--------------------------------
`wan2.2_ti2v_5B_fp16` (9,31 GiB) vem primeiro porque e **um arquivo so** que faz texto-para-video
sozinho: da para converter, renderizar e comparar sem depender de video-guia nem de um segundo
especialista. E o ciclo completo mais barato de uma familia.

`wan2.2_animate_14B` vem em seguida em DOIS arquivos -- BF16 e o `int8_convrot` do proprio
Comfy-Org -- porque e o confronto direto que esta bancada ja fez em quatro familias, agora na
familia Wan mais nova (Animate, modificada 9-13 ago 2026 segundo `lastModified` na API). Ter a
fonte E o adversario oficial no mesmo download evita descobrir depois que o nosso perdeu e nao
ter com o que comparar.

FICA DE FORA, e e escolha: `wan2.2_t2v_high_noise_14B` + `low_noise` sao um par de especialistas
MoE de 26,62 GiB CADA -- 53 GiB so na fonte, sem contar saida. Nao cabe junto com o resto.

GUARDA DE DISCO
---------------
Para antes de cada arquivo se o livre em F: nao cobrir o arquivo + 12 GiB de folga. Recusar e
mais barato que encher o disco no meio: hoje esta maquina ja chegou a 25 GB livres.
"""
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
DESTINO = r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models"
FOLGA = 12 * 2 ** 30

ARQUIVOS = [
    ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", 9999658848),
    ("split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors", 18413068672),
    ("split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors", 34549787368),
]

for remoto, esperado in ARQUIVOS:
    nome = remoto.rsplit("/", 1)[-1]
    alvo = os.path.join(DESTINO, nome)
    if os.path.exists(alvo) and os.path.getsize(alvo) == esperado:
        print(f"ja existe e bate: {nome}", flush=True)
        continue
    livre = shutil.disk_usage(DESTINO).free
    if livre < esperado + FOLGA:
        print(f"PARANDO antes de {nome}: {livre / 2**30:.1f} GiB livres nao cobrem "
              f"{esperado / 2**30:.1f} + 12 de folga. Nao e erro, e a guarda.", flush=True)
        break
    print(f"baixando {nome} ({esperado:,} B; {livre / 2**30:.1f} GiB livres) ...", flush=True)
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO, filename=remoto, local_dir=DESTINO)
    if os.path.abspath(p) != os.path.abspath(alvo):
        os.replace(p, alvo)
    n = os.path.getsize(alvo)
    print(f"  {nome}  {n} B  em {time.time() - t0:.0f}s  "
          f"{'OK' if n == esperado else f'TAMANHO DIFERE (esperado {esperado})'}", flush=True)
    if n != esperado:
        sys.exit(1)

# a arvore que o hf_hub_download recria ao lado
resto = os.path.join(DESTINO, "split_files")
if os.path.isdir(resto):
    shutil.rmtree(resto, ignore_errors=True)
cache = os.path.join(DESTINO, ".cache")
if os.path.isdir(cache):
    shutil.rmtree(cache, ignore_errors=True)

print("WAN_DOWNLOAD_OK", flush=True)
