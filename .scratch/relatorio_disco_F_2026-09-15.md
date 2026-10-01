# Disco F:, 2026-09-15 — o que saiu, o que ficou, e o que dá pra agrupar

Medido, não estimado. Espaço livre em F: antes **27,10 GiB**, depois **113,31 GiB** — a leitura do
driver, não a soma dos alvos; a soma dos alvos dá 86,08 e a diferença de 0,13 é folga de cluster.

## O que saiu, com a porta que autorizou cada um

| alvo | GiB | porta |
|---|---|---|
| `.git/lfs/objects` + `/tmp` | 27,14 | `git lfs ls-files --all` **vazio** (nenhum commit da árvore referencia LFS) **e** os 6 objetos provados cópia **sha256-idêntica** dos shards de `models/llm/Huihui-GLM-4.6V-Flash-abliterated-TEMP/`, que continuam no disco. 6 cópias, 0 únicos. |
| `git gc` | 0,59 | `.git` 671 → 67 MB; `fsck --connectivity-only` limpo; HEAD em `f34bb45`. |
| `F:\caches\pip` | 30,29 | só download. Os wheels **construídos** aqui somam 20 MiB e são todos puro-python (jieba, whisper, fairscale). Nenhum pacote instalado tocado. |
| `F:\caches\uv` | 11,98 | idem; `archive-v0` é hardlink, os venvs sobreviventes ficam intactos por construção. |
| `_backup_20260816_preupgrade` | 8,54 | decisão do dono: rollback do `python_embeded` de 16/08, **não refazível**. A stack andou para torch 2.13 desde então. |
| `venvs/ultravox311` | 4,62 | venv lateral Ultravox/TTS, sem relação com o ComfyUI. `venvs/comfymcp` **ficou** (é o do servidor MCP). |
| `ComfyUI/output` | 2,51 | 8084 de 8483 arquivos. Guardei 399: todo nome citado em arquivo versionado, mais os irmãos da mesma corrida. |
| `.scratch/te_*.pt` | 0,42 | quem os cita é a **própria sonda que os escreve** (`tools/probe_te_cadeado.py`, caminho de saída), não um documento que os use como evidência. |
| `F:\python`, `F:\tmp_verif` | 0 | `os.listdir` vazio no instante da remoção. |

**Duas armadilhas pegas antes de custar.** O `.flac` que acompanha o `ltx23av_bf16_av_00001_.mp4`
— citado no `HANDOFF_2026-09-14_FECHAMENTO.md` como prova — não compartilha o token `_av_` e teria
ido embora pela regra de prefixo; a regra de irmão (≥12 caracteres em comum no mesmo diretório)
salvou ele e os 249 quadros da mesma corrida. E `.scratch` **não foi reorganizado**: 153 arquivos
ali são versionados e ~50 caminhos estão citados em `CLAUDE.md`, `W4A4_PROGRESS.md` e nos tickets;
mover qualquer um cria link morto.

Conferido depois: `torch 2.13.0+cu130`, `cuda.device_count() == 2`, `safetensors 0.8.0`,
`transformers 4.57.6`, `comfy-kitchen 0.2.31` — todos importam.

## O que ficou de propósito

| alvo | GiB | por quê |
|---|---|---|
| `F:\_migracao` | 21,92 | `hv15_int8_convrot` (nunca publicado, nunca medido na saída) + `zimage_turbo_recuperado_bf16` (metadata diz `dequantized_from`). Fora de toda raiz declarada — o ComfyUI **não enxerga** nenhum dos dois. Reconvertíveis das fontes que estão no disco. |
| ~~`COMFY_PORTABLE\models\llm\Huihui-GLM-4.6V-...-TEMP`~~ | ~~19,19~~ | **APAGADO em 2026-09-16 21:24, por ordem do dono.** Ver o adendo no fim. |
| `calib` | 18,80 | ativações de calibração; refazer exige amostragem real na GPU. |
| `venvs/comfymcp` | 0,18 | em uso pelo servidor MCP. |

## Mapa da raiz de F: — nada foi movido

66 entradas. Ordenado pelo que é mais antigo, que é onde mora o agrupável.

| pasta | GiB | última escrita | git | leitura |
|---|---|---|---|---|
| `COMFY_PORTABLE` | 1112,96 | 15/09 | sim | esta bancada |
| `tts_lab` | 182,33 | 14/09 | não | vivo |
| `SteamLibrary` | 149,32 | 19/08 | — | jogos |
| `qwen38-27b-rtx3090` | 52,60 | 01/09 | sim | vivo |
| `IA` | 43,55 | 25/08 | não | |
| `frankestein-ugc` | 36,12 | 10/08 | não | parado há 5 semanas |
| `PlexData` | 30,10 | 26/08 | não | |
| `tts-ab-test` | 27,25 | **21/06** | não | **parado há ~3 meses**; sucedido pelo `tts_lab` |
| `cortex` | 21,99 | 15/09 | sim | vivo |
| `_migracao` | 21,92 | 01/09 | não | ver acima |
| `cortiq` | 20,70 | 21/08 | não | o handoff do cortiq aponta para `cortiq-cmf` |
| `build_torch` | 20,55 | 14/08 | não | árvore de build do torch, 233 mil arquivos |
| `ptbr-audio-lab` | 10,71 | 13/08 | não | |
| `zhao-hub-data` | 10,27 | 01/09 | não | dado do ERP antigo, guardado de propósito |
| `ugc-studio` | 9,78 | 14/08 | não | |
| `cortex-snapshots` | 9,63 | **31/07** | não | snapshots antigos do cortex |
| `lf_embeded` | 9,42 | 02/07 | não | python embarcado de outro projeto |
| `BACKUP DO SSD DE 256` | 9,18 | **13/04** | não | backup de 5 meses |
| `Local_Studio` | 7,76 | 29/08 | não | 373 mil arquivos |
| `models` | 6,31 | 31/07 | não | LFM2.5 distill |
| `estudo` | 4,56 | 10/08 | não | |
| `Area-51` | 2,90 | **02/11/25** | não | **10 meses parado** |
| `hf-cache` | 2,60 | 14/09 | não | vivo; contém credencial do HF, não tocar |
| `ls-main-f` | 2,50 | 30/08 | sim | |
| `cortiq-cmf` | 2,08 | 21/08 | sim | o que o `CORTIQ_LTX25_HANDOFF.md` manda ler |
| `textgen-portable-4.9` | 1,91 | 24/05 | não | |
| `zhao-bench-282` / `-D` / `-data` | 1,11 | 21–22/08 | não | três benches do mesmo dia |
| `OSINT-PROPRIO` | 0,45 | 06/06 | não | |
| `SIMNext` | 0,26 | 09/01 | não | |
| `COMFY_PORTABLE_UPDATE_BACKUP` | 0,14 | 17/08 | não | rollback do upgrade de 17/08 |
| `tmp-nft`, `nft-probe-base`, `pb_sync`, `limpa_org_local`, `test`, `cortex-treinamento`, `caches` | ~0 | 06/08–15/09 | | sobra de sonda |

**O que eu agruparia, se você mandar** (nada disso foi feito): um `F:\_arquivo\` com
`tts-ab-test`, `cortex-snapshots`, `Area-51`, `BACKUP DO SSD DE 256`, `lf_embeded`,
`textgen-portable-4.9`, `SIMNext`, `OSINT-PROPRIO` e as sobras de sonda de ~0 GiB — 60,7 GiB em
oito pastas paradas há 1 a 10 meses. Mover não libera byte nenhum; só tira 8 entradas da raiz. E
`build_torch` (20,55 GiB, 233 mil arquivos) é árvore de build: reconstruível, mas é decisão sua.

**Por que não movi nada.** Mover um diretório de projeto quebra caminho absoluto em script, atalho,
`.env` e config de quem aponta para ele — e nada aqui me diz quem aponta. Uma pasta parada há três
meses pode ser insumo de um `.bat` que roda uma vez por trimestre.

## Não coberto

A busca de nome em `ComfyUI/output` é por **texto literal** nos arquivos versionados: um documento
que cite um render por glob ou por descrição, sem o nome, não teria sido visto. Os tamanhos da raiz
vêm de `robocopy /l` com `/xj`, então junction não é seguida e hardlink conta em cada caminho — o
número do `uv` antes de apagar era, por isso, um teto e não um ganho garantido. Não olhei dentro de
`tts_lab`, `IA`, `PlexData`, `Local_Studio`, `frankestein-ugc` nem `SteamLibrary`: são do dono e o
pedido era limpar, não auditar projeto alheio. O `.git/lfs` foi varrido em **todo** o F: — era o
único.

## Adendo 2026-09-16 - o GLM-4.6V: havia TRES copias, e a do Comfy nao era lida por ninguem

| onde | qual modelo | GiB | quem le |
|---|---|---|---|
| `P:/IA - Linux/Modelos - LM/Huihui/Huihui-GLM-4.6V-Flash-abliterated` | abliterated | 19,19 | **as corridas do unsloth**: o `adapter_config.json` de 01/09 10:22 e de 10/09 19:37 grava essa UNC como `base_model_name_or_path` |
| `C:/Users/joaoz/projetos/project_quant_merge_frankestein/models/zai-org__GLM-4.6V-Flash` | zai-org **original**, outro modelo | 19,19 | as corridas de 11/09 em diante, inclusive as de 16/09 |
| `F:/COMFY_PORTABLE/models/llm/Huihui-...-TEMP` | abliterated | 19,19 | ninguem - **apagada** |

(caminhos com barra normal para nao depender de escape; na maquina sao barras invertidas)

**Tres medicoes independentes** disseram que a do Comfy estava morta: (1) o `adapter_config.json`
das duas corridas do abliterated nomeia a UNC da NAS; (2) os arquivos pequenos da copia da NAS -
que nenhum script meu leu - tem acesso real em 09/09 19:31, 10/09 08:10 e 18:03, 12/09 21:29 e
16/09 08:13, enquanto os mesmos arquivos da copia do Comfy marcam apenas 15/09 20:04, que e o
rastro da minha propria varredura; (3) nenhum script, config ou log em `tools/`, `.scratch/`,
`ComfyUI/user/`, `F:/limpa_org_local`, `F:/IA` ou `C:/Users/joaoz/.unsloth` nomeia o caminho local.

Prova de identidade: **6/6 sha256 iguais** nos cinco shards e no tokenizer, 15/15 arquivos com mesmo
nome e tamanho. O HF **nao** tem esse modelo - o unico repo com "GLM" no nome
(`glm-4-voice-bridge-GLM-4.6V-Flash-alpha`) sao 2,5 MB de pontes r16, logs e scripts, zero peso.

**Erro meu, registrado:** o carimbo de ultimo acesso da copia do Comfy foi destruido pela minha
varredura de 15/09 20:04 - o NTFS registra acesso nesta maquina (`DisableLastAccess = 2`), o dado
existia, e eu o apaguei ao medir. A resposta so apareceu porque os **arquivos pequenos** da copia da
NAS sobreviveram intactos (meu hash so leu os maiores que 1 MB). **Ler e escrever, num sistema que
carimba acesso.** Hashear uma arvore antes de perguntar quando ela foi usada pela ultima vez queima
a propria evidencia.

**Segundo erro, registrado:** a porta P3 do script de remocao (mtime da NAS dentro da janela do
download de 11/08) usava uma janela de epoch errada - 1786000000 a 1786100000, quando 11/08 22:16 e
1786497360. Ela marcou os 15 arquivos como "fora" e **nao verificou nada**; como nao abortava por
desenho, a remocao correu sobre P1, P2, P4 e P5. O mtime da NAS ficou sem conferencia. Uma porta que
sempre falha e nunca barra e pior que porta nenhuma: ocupa o lugar de uma que funcionaria.

Liberado: **19,19 GiB**. F: passou de 113,31 para 132,25 GiB livres. A pasta
`F:/COMFY_PORTABLE/models/` ficou vazia (o GLM era todo o conteudo) e foi deixada no lugar.
