# LTX 2.5, o que já rodou nesta máquina — registro de execução

Atualizado 2026-09-12. Tudo aqui é **executado**, não traçado.

## O braço original funciona, e precisa das duas placas

`ltx-2.5-22b-distilled-transformer-bf16.safetensors`, 42.018.190.584 B (39,13 GiB),
com o encoder `gemma4-12b-with-proj-ltx-2.5-bf16` (26.263.858.182 B) e o
`ltx-2.5-video-vae-bf16`. Ambos byte a byte iguais aos do `Lightricks/LTX-2.5` no Hub.

**Primeira tentativa, sem DisTorch2: `CUDA error: out of memory`**, levantado de dentro de
`torch.cuda.mem_get_info` chamado por `model_management.free_memory` → `unload_all_models`.
Não foi uma alocação infeliz: a placa esgotou. 39,13 GiB contra 24 GiB não cabe nem com o
offload normal do ComfyUI.

**Com DisTorch2 espalhando blocos, roda.** Alocação aceita e confirmada no log do próprio
pacote:

```
[MultiGPU DisTorch V2] Full allocation string: cuda:1,6.0gb;cpu,*#cuda:0;4.0;cpu
Model LTXAV prepared for dynamic VRAM loading. 40048MB Staged.
```

São **6 GiB** doados pela 3080 Ti, não 12: o embedding do cortex mora nessa placa com
~2,3 GiB medidos e não pode ser expulso.

## Medição

| corrida | quadros | resolução | passos | tempo | s/quadro |
|---|---|---|---|---|---|
| fumaça BF16 | 49 (1,96 s) | 512×512 | 3 | **717,8 s** | 14,65 |

Tempo lido do `/history` do worker (`execution_success` − `execution_start`), não cronometrado
por fora. Status `success`, 49 PNGs.

`3` passos porque é o destilado: sigmas manuais `0.909375, 0.725, 0.421875, 0.0`, cfg 1.0 —
o regime do workflow de aceitação que já rodava aqui, não um regime inventado.

![tira do braço BF16](tira_smoke_bf16_49q.png)

Quadros 1, 17, 33 e 49. Farol, ondas quebrando, gaivotas — coerente e com movimento real.

## A armadilha que custou uma corrida

Com `COMFYUI_MGPU_DISABLED` diferente de 1, o ComfyUI-MultiGPU sobe **um worker por placa** em
portas escolhidas em tempo de execução e encaminha o trabalho. O resultado aparece no
`/history` **do worker**; a fila do servidor principal fica vazia enquanto a GPU está a 85%.
Isso é indistinguível de "já terminou" e de "nunca começou". `tools/ltx_video.py` e
`tools/qwen_edit_test.py` agora leem as portas dos logs do pacote e olham nos dois lugares.

## O que NÃO está coberto

- Uma resolução (512), um prompt, uma semente, um sampler, um `frame_rate`.
- O **áudio não foi decodificado**. O latente de áudio entra porque o grafo concatena, mas só o
  ramo de vídeo é decodificado.
- Nada aqui compara qualidade entre formatos ainda — só estabelece que o original roda e
  quanto custa.
- O `s/quadro` inclui carga e staging de 40 GB. Não é custo marginal por quadro, e **não
  escala linearmente**: uma corrida de 249 quadros amortiza a carga sobre 5,08x mais trabalho.

---

## Duas armadilhas que custaram 36 minutos, registradas para não custarem de novo

### 1. O worker do MultiGPU ignora o `compute_device` do nó

Com `COMFYUI_MGPU_DISABLED` diferente de 1, o pacote sobe um worker por placa e distribui os
trabalhos **por rodízio**. A primeira corrida foi para o worker da GPU 0; a segunda foi para o
worker da **GPU 1** — a 3080 Ti de 12 GB — mesmo com o nó pedindo `compute_device: cuda:0`.

O resultado, lido do log do worker:

```
mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
torch.AcceleratorError: CUDA error: out of memory
```

39,13 GiB numa placa de 12 GB. E o mais caro: a fila continuou dizendo `rodando: 1` **depois**
do OOM, com a GPU 0 ociosa em 659 MiB — 36 minutos parecendo progresso.

**A alocação da DisTorch2 não protege contra isso**, porque ela decide onde os *blocos* moram
dentro do processo que já foi escolhido. Quem escolhe o processo é o rodízio, antes.

Correção: `COMFYUI_MGPU_DISABLED=1`. Os nós `*DisTorch2MultiGPU` continuam registrados e
funcionando — a variável controla o *spawn de workers*, não o registro dos nós.

### 2. Matar o servidor pelo arquivo de pid pode matar o processo errado

O `.pid` foi sobrescrito por um lançamento posterior, então `taskkill` matou um número que já
não era o servidor. O antigo continuou dono da porta 8190, e o teste de saúde respondeu
**`ONLINE após 5s`** — rápido demais para o ComfyUI, que leva ~60 s para subir. Essa velocidade
foi o sinal, e quase passou batido.

O que decide é quem **possui a porta**, não quem responde nela:

```powershell
Get-NetTCPConnection -LocalPort 8190 -State Listen | Select-Object OwningProcess
```

Deu `17964` quando o esperado era `34100`. Confirmar isso antes de acreditar que o servidor foi
reiniciado.

## 2026-09-13 — os três braços de novo, com ÁUDIO, e o servidor que morreu no caminho

O dono apontou que o LTX gera vídeo E áudio e que a comparação publicada media só o vídeo. Os três
braços foram rerenderizados com `tools/ltx_video.py` (que substitui o `ltx25_video.py`): PNG por
quadro, FLAC do áudio, MP4 com faixa. Mesma semente, mesmo encoder, mesmos sigmas.

**Os quadros voltaram pixel a pixel iguais aos da primeira rodada**, nos três braços (MAE 0,0 nos
quadros 1/63/125/187/249), atravessando um reinício de servidor e, no BF16, outro disco. O hash
dos PNGs difere só pelo metadado (o grafo agora tem os nós de áudio). Logo os números de vídeo são
os mesmos; os de áudio são novos (`bench/ltx25/av/comparacao_av.json`):

```
braço                 MAE   PSNR   SSIM   | log-mel L1   SNR      lag    conv   RMS
int8 Lightricks      4,10  29,71  0,941  |   0,041     11,2 dB   0 ms  0,124  -38,8 dBFS
W4A8 nosso           7,81  25,39  0,895  |   0,120      3,4 dB   0 ms  0,311  -38,3 dBFS
controle: silêncio                       |   6,980      0,0 dB
controle: ruído branco (mesmo RMS)       |   1,471     -3,0 dB          1,097
```

O áudio ordena igual ao vídeo e por margem maior: 1,9x mais longe no quadro, 2,9x no som. Nenhum
braço mudou de nível nem deslocou no tempo.

**O braço BF16 derrubou o servidor na primeira tentativa.** `Windows fatal exception: access
violation` em `torch/storage.py __getitem__`, chamado de `comfy/utils.py:136 load_torch_file`
dentro do `UNETLoaderDisTorch2MultiGPU` — o mmap do arquivo de 39 GiB, que o resolvedor de nomes
lia de **D: (SMB)**, com ~24 GiB de RAM livre (uma conversão de 43 GiB e um download de 23 GiB
corriam ao mesmo tempo). O mesmo arquivo existe em W: (byte a byte igual); um **hardlink com outro
nome** (`ltx-2.5-22b-distilled-transformer-bf16_W.safetensors`, mesmo inode, nada copiado nem
movido) faz o loader ler de W:. Com 40 GiB livres e sem conversão concorrente: 769,6 s, `cpu,40gb`,
sucesso, e pixel-idêntico.

**CORREÇÃO (22:08 do mesmo dia): W: NÃO é disco local.** `net use` lista `W: \\192.168.3.40\zfe`
— compartilhamento SMB, como P: e D:. Eu tinha deduzido "local" de um grep de `net use` que só
procurava D:. Logo as duas cargas do BF16 foram mmap por SMB: uma morreu, uma sobreviveu. O 2.3
acrescentou TRÊS mortes com a mesma assinatura (`access violation` em `torch/storage.py
__getitem__`, dentro do `get_tensor` do `load_torch_file`), em arquivos de 39–43 GiB, com 32–44 GiB
de RAM livre — enquanto um processo Python nu percorreu o mesmo arquivo de 39 GiB em 4 s. O
redirecionador ficou como suspeito por algumas horas, e como julgamento; o parágrafo seguinte é a
medição que o substituiu. Os únicos discos NTFS locais são C: e F:. Lição que
fica: o resolvedor do ComfyUI devolve o PRIMEIRO caminho do yaml que tem o nome, e a ordem do yaml
decide de qual VOLUME um mmap de 39 GiB sai — e isso não aparece em log nenhum.

**CORREÇÃO DA CORREÇÃO (2026-09-14, 00:18–00:30, MEDIDO): não é o SMB, é COMMIT.** Três probes num
processo nu, sem ComfyUI, lendo os contadores de commit do sistema antes e depois de cada chamada
(`scratchpad/probe_commit_mmap.py`, `probe_safeopen_trace.py`, `probe_double_map.py`), no mesmo
arquivo de 39,13 GiB:

```
chamada                                                   cobrança de commit
safetensors.safe_open(framework="pt")                     +40,8 GiB ao abrir, antes de ler tensor algum
   (duas views copy-on-write do mesmo arquivo: memmap2 para o cabeçalho e
    torch.UntypedStorage.from_file(shared=False) para os dados; +80,2 GiB enquanto as duas vivem)
torch.empty dos parâmetros do modelo                      +40,7 GiB a mais
mmap somente-leitura (numpy, loader GGUF)                    0
UntypedStorage.from_file(shared=True)                        0
```

O Windows cobra uma view copy-on-write pelo tamanho inteiro no ato do mapeamento, então o caminho
normal do ComfyUI precisa de 2x o arquivo em commit só para abrir e 3x para construir o modelo. O
teto desta máquina é 124,8 GiB (63,6 GiB de RAM + pagefile de 61 GiB gerido pelo sistema) com ~70 GiB
já comprometidos por outros processos. Quando a cobrança força o pagefile a crescer, a view às vezes
volta com o limite aumentado e a cobrança NÃO feita, e a primeira leitura por ela dá access violation
em `torch/storage.py __getitem__` — a assinatura exata do servidor (`safe_open` + primeiro
`get_tensor` em W:, 3 de 3; o mapeamento duplo feito à mão em C:, 1 de 1; a mesma chamada sobrevive
em outras corridas, que é por que uma carga do BF16 em sete deu certo). As duas mortes dentro de
`nn.Linear.__init__` (o `torch.empty` do modelo) têm a mesma assinatura e não foram reproduzidas
isoladas. O volume nunca importou. O braço BF16 do 2.3 foi renderizado SEM o leitor de safetensors:
os mesmos bytes BF16 num GGUF (`tools/safetensors_to_gguf_bf16.py`), lido por memmap
somente-leitura pelo `UnetLoaderGGUF` — bytes, peso dequantizado e saída do Linear conferidos
idênticos em 12 camadas (`tools/probe_gguf_bf16_equivalence.py`). Regra que fica: nesta máquina, não
abrir por `safe_open` arquivo maior que metade do commit livre; encoder fora do processo
(`--encode-only` / `--cond-from`).

Os tempos da tabela do card continuam sendo os da primeira rodada (rede ociosa). Nesta rodada o
int8 levou 801,5 s e o W4A8 511,5 s porque a carga saiu do NAS durante o download e a conversão —
tempo de carga, não de amostragem, e por isso não substitui os números publicados.
