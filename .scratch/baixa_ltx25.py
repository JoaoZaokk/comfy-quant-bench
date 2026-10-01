"""Baixa o transformer BF16 do LTX 2.5, que e a fonte sem a qual nada da familia se mede.

POR QUE ESTE ARQUIVO PRIMEIRO
-----------------------------
Nenhum DiT de LTX **sem quantizacao** existe nesta maquina -- confirmado por dois caminhos em
2026-09-12. O maior LTX local e `ltx-2.3-22b-dev-fp8`, que ja e fp8, e do 2.5 so ha as tres
builds quantizadas do riftcast. Sem a fonte BF16 nao ha calibracao em ativacao real, nao ha erro
por camada, e nao ha o braco "original" que o goal pede para o video de 10 s.

E ele e necessario sob QUALQUER leitura da decisao de disco pendente: se o dono liberar espaco
para o encoder BF16 tambem, este arquivo entra igual; se optar por usar o encoder w4a8 que ja
esta no disco, este arquivo entra igual. Entao baixar agora nao antecipa a decisao dele.

DISCO
-----
39,13 GiB contra 58 GB livres. Sobra ~19 GB, que cobre uma saida W4A8 (~11 GiB) com folga
apertada. O conversor recusa em vez de encher o disco, entao o modo de falha e uma recusa e nao
um sistema travado.

`ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` (20,03 GiB) -- o braco adversario, e desta
vez publicado pela PROPRIA Lightricks -- fica de fora por disco. E escolha registrada, nao
esquecimento: com 19 GB livres depois deste download ele nao cabe junto com a nossa saida.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "Lightricks/LTX-2.5"
DESTINO = r"F:\COMFY_PORTABLE\ComfyUI\models\diffusion_models"
ARQUIVOS = [
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", 42018190584),
]

for remoto, esperado in ARQUIVOS:
    nome = remoto.rsplit("/", 1)[-1]
    alvo = os.path.join(DESTINO, nome)
    if os.path.exists(alvo) and os.path.getsize(alvo) == esperado:
        print(f"ja existe e bate: {nome}", flush=True)
        continue
    print(f"baixando {nome} ...", flush=True)
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO, filename=remoto, local_dir=DESTINO,
                        local_dir_use_symlinks=False)
    # hf_hub_download com local_dir recria a arvore; move para a raiz da pasta
    if os.path.abspath(p) != os.path.abspath(alvo):
        os.replace(p, alvo)
    n = os.path.getsize(alvo)
    ok = "OK" if n == esperado else f"TAMANHO DIFERE (esperado {esperado})"
    print(f"  {nome}  {n} B  em {time.time() - t0:.0f}s  {ok}", flush=True)
    if n != esperado:
        sys.exit(1)

print("LTX25_DOWNLOAD_OK", flush=True)
