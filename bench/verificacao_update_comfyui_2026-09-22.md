# Verificacao: da para atualizar o ComfyUI, e o que ele diz do Qwen-Image-2.1

Pedido dele: verificar se da para atualizar, fazer o backup, e testar o suporte ao Qwen-Image-2.1
chamando o upstream **por fora, manualmente**, sem reinventar a roda. **Nada foi atualizado.** A
instalacao nao foi tocada em nenhum passo.

---

## 1. A resposta que ele queria: SIM, o upstream reconhece o Qwen-Image-2.1

    comfy.model_detection.model_config_from_unet, ComfyUI upstream v0.37.0-11-gb33e2b55
      Qwen-Image-2.1  prefixo ''                        -> QwenImage21
      Qwen-Image-2.1  prefixo 'model.diffusion_model.'  -> QwenImage21
      CONTROLE  Qwen-Image-2512                         -> QwenImage
      CONTROLE  FLUX.2-klein-4B                         -> Flux2

Na instalacao atual (0.33.0) o mesmo arquivo da **`None` nos dois prefixos** -- medido ontem. Os dois
controles continuam detectando, entao o arnes nao esta inventando o resultado.

**Como foi testado, sem tocar a instalacao:** `git archive origin/master` num diretorio de rascunho
(leitura pura, 51 MiB) e `comfy-aimdo 0.5.5` por `pip install --no-deps --target` noutro diretorio de
rascunho. Os dois entram so no `sys.path` daquele processo; conferido depois que o `site-packages`
segue com **comfy-aimdo 0.4.13**.

E o suporte veio com as melhorias que ele previu: alem do `6bfaacc6 feat: Qwen-image 2.1 support
(CORE-423) (#16400)`, ja entraram `4d7e61b7` (conserto), `2f7c6d47` (logica de local do cache KV) e
`1d61dcc3` (compile dos blocos do transformer).

## 2. O custo da atualizacao e pequeno, e NAO toca o Torch

`v0.33.0-19-gc1739380` (2026-08-18) contra `v0.37.0-11-gb33e2b55`: **209 commits**, e o repo local nao
tem nenhum commit proprio a frente do upstream.

Diferenca de `requirements.txt`, **por versao**:

    pacote                       instalado    upstream pede   veredito
    av                           18.0.0       >=17.0.0        JA SATISFEITO
    comfy-aimdo                  0.4.13       ==0.5.5         precisa subir  <- bloqueio duro
    comfy-kitchen                0.2.31       ==0.2.35        precisa subir  <- O RISCO
    comfyui-frontend-package     1.49.6       ==1.53.6        precisa subir  (so UI)
    comfyui-embedded-docs        0.5.10       ==0.5.12        precisa subir  (so docs)
    comfyui-workflow-templates   0.11.43      ==0.11.68       precisa subir  (so templates)

**Torch, CUDA, numpy, safetensors, transformers, triton: nenhum muda.** Isso e o oposto do que
quebrou esta instalacao antes, quando o `update_comfyui_and_python_dependencies.bat` levou o Torch para
2.13.0+cu130 e a SageAttention parou por DLL. Aqui a regra de nao fazer upgrade em massa **nao precisa
ser violada**.

**O bloqueio duro e o `comfy-aimdo`, e ele e real, nao cosmetico.** Tentei rodar o upstream com o
0.4.13 e ele morre em cadeia: primeiro `ModuleNotFoundError: comfy_aimdo.storage` (que o
`comfy/storage.py` novo importa), e com aquele stubado, `comfy_aimdo.malloc_graph` (que o
`comfy/model_prefetch.py` novo importa). Stub de um resolve, stub de dois e teimosia: instalei o 0.5.5
de verdade num diretorio separado e ai tudo importou.

**O RISCO e o `comfy-kitchen` 0.2.31 -> 0.2.35.** E o registry que decide se `convrot_w4a4_linear`
resolve para `comfy_kitchen.backends.cuda.*`, e **todo numero que esta bancada publicou depende dele**.
Subir sem re-rodar o preflight e `probe_quant_dispatch` em cada build seria publicar numeros de um
despacho que ninguem conferiu. [JULGAMENTO] e a unica peca desta lista que eu nao subiria sem a placa
livre para reconferir.

## 3. Os nossos 5 patches locais: 3 aplicam limpo, 2 conflitam

Nao sao cosmeticos -- sao **a liberacao dos dois cadeados do text encoder** que esta bancada mediu,
mais o conserto do `symmetric_patchifier` do LTX e a flag `--disable-quantized-text-encoder`. 37
insercoes, 12 delecoes, em 5 arquivos.

    comfy/cli_args.py                             APLICA LIMPO
    comfy/ldm/lightricks/symmetric_patchifier.py  APLICA LIMPO
    comfy/sd1_clip.py                             APLICA LIMPO
    comfy/ops.py                                  CONFLITA  (upstream mexeu perto de :1287)
    comfy/sd.py                                   CONFLITA  (upstream mexeu perto de :265)

O upstream mudou muito nesses arquivos: `ops.py` +130/-, `sd.py` +107/-, e `sd.py` e **tambem um dos 11
arquivos que o commit do Qwen-Image-2.1 toca**. Entao o conflito nao e evitavel escolhendo commits.

**E eu conferi se o patch ainda e necessario, porque "nao reinventa a roda" vale para o meu codigo
tambem.** O upstream v0.37 mantem **os dois cadeados**: `full_precision_mm=True` fixo em
`sd1_clip.py:114` e `set_model_compute_dtype(torch.float32)` incondicional em `sd.py:274`, com o mesmo
comentario de sempre. Nao ha flag nova de text encoder quantizado. **O patch segue sendo nosso e segue
sendo necessario** -- reaplica-lo a mao nos dois arquivos e trabalho de leitura, nao de adivinhacao,
e o patch do backup e o gabarito.

## 4. Backup feito

`F:\COMFY_PORTABLE_BACKUP_2026-09-22_pre_v0.37`, 45 MiB:

    ComfyUI_core_HEAD.tar               46.725.120 B   o core rastreado em HEAD
    patches_locais_nao_commitados.patch      6.695 B   os 5 arquivos, conferidos com --check --reverse
    pip_freeze_2026-09-22.txt               12.341 B   o estado exato dos pacotes
    constraints_local.txt / requirements_local.txt     os pins
    custom_nodes_commits.txt                 5.111 B   nome + sha + remote de cada pack com .git
    custom_nodes_inventario.txt / git_status.txt / HEAD.txt

O `--check --reverse` do patch passa, isto e: **o patch descreve exatamente o estado atual do disco**,
nao uma aproximacao. E existe um backup anterior, de 2026-07-23, em
`F:\COMFY_PORTABLE_UPDATE_BACKUP` (142 MiB), da ultima atualizacao controlada -- com procedimento
gravado na sessao `65225d29` do cortex, inclusive o robocopy de rollback do MultiGPU legado.

## 5. Docker e runtime, pelo cortex: o que existe e o que NAO existe

Procurado no cortex por quatro consultas e pelo indice de friccao. **Nao existe nenhuma sessao de
ComfyUI rodando dentro de container.** O que existe e util:

- **A GPU passa para o container nesta maquina, e isso foi verificado, nao suposto.** Sessao
  `f6d1e2f9` (2026-06-23): verificacao nao-destrutiva rodando so os imports dentro da imagem --
  `IMPORTS_OK torch 2.9.1+cu`, `sgl_kernel`, `flashinfer`, e "o container enxerga a placa". Foi
  deliberadamente **nao** subido o container completo para nao brigar com o shim WSL, que usava a
  mesma porta 8123 e a mesma GPU 0.
- **O sofrimento com docker que ele lembra e do projeto de servir LLM, nao do ComfyUI.** Sessoes
  `413fcfe3` / `dc55ec35` / `aec6294c` ("W4A4 - LLM"): compose reescrito com um perfil por servico e
  porta propria, e as armadilhas medidas la -- **servico sem perfil entra em toda invocacao**, entao
  `--profile awq_dual up` subiu o `awq` junto e dois containers de 18 GB brigaram pela mesma placa; e
  `/opt/qwen38/docker/` nao existia na imagem porque o compose nao montava o que o README dizia.
- **O estado do docker desta maquina** esta no `CLAUDE.md` e foi medido em 2026-09-20: 20 containers,
  4 rodando, `docker_data.vhdx` com 318,86 GiB **genuinamente cheio** (313,5 GB de conteudo real), 157,4
  GB reclamavel -- **podar antes de compactar**, e `wsl --shutdown` derruba os 4 que rodam, incluindo o
  `gestao-db`.

[JULGAMENTO] docker nao ajuda no problema de hoje. A pergunta do Qwen-Image-2.1 ja foi respondida por
fora, num processo, em segundos, sem container. Docker passaria a valer se a decisao fosse **rodar as
duas versoes do ComfyUI em paralelo** de forma duravel -- e nesse caso o custo real nao e a imagem, e
os ~600 GiB de modelos que teriam de ser montados de fora, mais a briga por porta e por GPU que aquela
sessao ja registrou.

## 6. Um erro do meu instrumento, no meio desta verificacao

Minha primeira comparacao de `requirements.txt` devolveu **"EXIGENCIAS NOVAS: []"** e eu quase reportei
que a atualizacao nao pedia pacote nenhum. Ela comparava **conjuntos de NOMES**, e os 6 pacotes que
mudaram estao nos dois arquivos -- com versoes diferentes. Um diff de nomes **nao ve bump de versao por
construcao**. Refeito comparando o pin, e e de la que sai a tabela da secao 2.

## Nao coberto

- **Nada foi atualizado, nada foi instalado no `site-packages`.** O `comfy-aimdo 0.5.5` foi para um
  diretorio de rascunho com `--no-deps --target`, e a instalacao segue no 0.4.13, conferido.
- **Detectar nao e renderizar.** O teste da secao 1 resolve arquitetura por shape em tensores
  `device='meta'`: nenhum peso carregado, nenhum kernel, nenhuma amostragem, **GPU nao tocada**. O
  text encoder do 2.1 (Qwen2.5-VL, 16,7 GiB) nao foi baixado e o VAE nao foi exercitado.
- **Os custom nodes nao foram avaliados** contra a v0.37. Sao 90+ pacotes, e uma atualizacao de core
  que quebra node nao aparece em nenhuma medicao acima.
- O `comfy-kitchen 0.2.35` **nao foi exercitado**: nao sei se o registry dele resolve os mesmos ops.
  Isso exige a placa.
- A reaplicacao dos 2 patches em conflito **nao foi feita nem testada**; sei que conflitam e onde.
