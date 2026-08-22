# Artefatos nossos dentro de `ComfyUI/`

Levantamento para o ticket `issues/16-artefatos-nossos-dentro-de-comfyui.md`. Método: comandos
executados (git, listagem, diff, grep de conteúdo), não leitura de nomes de arquivo. Onde a
conclusão vem de LER código em vez de rodá-lo, isso está dito explicitamente no item.

## Como o teste de "nosso vs upstream vs órfão" foi feito

Três fatos, todos por execução:

```
git -C ComfyUI log --oneline -5          -> HEAD em c1739380 (v0.33.0 + patches, 2026-08-20)
git -C F:\COMFY_PORTABLE ls-files | grep "^ComfyUI/"   -> ZERO arquivos. A bancada não rastreia
                                                          nada dentro de ComfyUI/, ponto.
grep "/user/\|/custom_nodes/" ComfyUI/.gitignore       -> ambos ignorados por inteiro:
                                                          "/user/" e "/custom_nodes/" (linha 8
                                                          e linha 20 do .gitignore upstream)
```

Consequência: **qualquer arquivo nosso salvo dentro de `ComfyUI/user/` ou
`ComfyUI/custom_nodes/`, sem também existir em algum caminho que a bancada rastreia, é órfão dos
dois repositórios ao mesmo tempo** — nem o checkout upstream vê (ignorado), nem a bancada vê (fora
da árvore que ela rastreia). `git status --short` nunca vai listar esses arquivos como
modificação nem como untracked comum; eles saem como `!!` (ignorado) se você pedir
`--ignored=matching`, o que foi feito para confirmar.

## O que `update_comfyui.bat` realmente faz com esses arquivos (LIDO, não executado)

Não rodei o update — a regra do ticket proíbe qualquer coisa que mexa em GPU/servidor, e
atualizar o ComfyUI está fora do que este ticket pediu de qualquer forma. Isto é leitura de
`update/update.py` (pygit2), não medição:

- `repo.stash(ident)` roda com os defaults do pygit2: `include_untracked=False`,
  `include_ignored=False`. Arquivos ignorados (como tudo em `user/` e `custom_nodes/`) **nunca
  entram no stash**, entram ou não é irrelevante aqui porque nem chegam a ser considerados.
- O pull subsequente (`checkout_tree` / `merge`) só escreve arquivos que estão na árvore git do
  remoto. Um arquivo nosso com nome que o upstream nunca vai usar (`LTX25-int8-acceptance-v2.json`,
  `comfy_convrot_native/`) não tem como colidir com um path que o merge decide escrever.
- Não há `git clean`, `rm -rf` nem remoção de diretório em `update.py` em lugar nenhum.

Conclusão da leitura: **rodar `update_comfyui.bat` sozinho não apaga esses arquivos.** O risco
real não é o update script — é qualquer coisa mais bruta que ele não faz mas que uma reinstalação
faria: `git clean -dfx` dentro de `ComfyUI/`, apagar e reclonar o checkout, restaurar de um backup
de disco que não inclua esses caminhos, ou qualquer fluxo de "começar do zero". Nesse cenário, um
arquivo que só existe ali dentro **desaparece sem deixar rastro em nenhum dos dois repositórios**,
porque não há segunda cópia em lugar nenhum. Essa é a frase que o critério de fechamento pede: o
risco de perda, escrito.

---

## Item por item

### 1. `ComfyUI/user/default/workflows/LTX25-int8-acceptance-v2.json`

- **Onde mora hoje**: `ComfyUI/user/default/workflows/`, 14 873 bytes, modificado 2026-08-19 11:59.
- **É nosso?** Sim. Confirmado por conteúdo (`python_embeded\python.exe -s`, `json.load` +
  busca de substring): contém `seed 1234` e `euler`, batendo com os parâmetros descritos no
  próprio ticket (512×512, 49 frames, 3 passos, cfg 1.0, euler, seed 1234, 25 fps). Referenciado
  por nome em `W4A4_HANDOFF.md` e `W4A4_PROGRESS.md` como o workflow da comparação — confirmado
  por `grep`.
- **Órfão dos dois repos**: sim (`/user/` é ignorado por inteiro no `.gitignore` do ComfyUI;
  bancada não rastreia nada em `ComfyUI/`).
- **Risco de perda num update**: baixo no `update_comfyui.bat` sozinho (ver seção acima); **alto**
  em qualquer reinstalação/`git clean`/restauração de backup que não preserve `user/` — porque não
  existe segunda cópia em lugar nenhum hoje. Sem ele, a comparação end-to-end (o próprio ticket
  diz isso) não se reproduz.
- **Decisão recomendada**: trazer uma cópia rastreada para a bancada (ex.: um diretório tipo
  `workflows/` no repo raiz), mantendo o original onde está para o ComfyUI continuar carregando-o
  normalmente pela UI. Não movido nesta rodada — decisão do dono.

### 2. `ComfyUI/user/default/workflows/LTX25-int8-acceptance.json` (v1)

- **Onde mora hoje**: mesmo diretório, 15 016 bytes, modificado 2026-08-19 11:54 (5 min antes do
  v2).
- **É nosso?** Sim, mesmos marcadores (`seed 1234`, `euler` presentes).
- **Diferença de papel**: `grep -n "LTX25-int8-acceptance" W4A4_PROGRESS.md W4A4_HANDOFF.md`
  filtrando fora as ocorrências de `-v2` acha só **uma linha**, um log de
  `validate_workflow` (`valid: true errors: 0 warnings: 0`) — não é citado como o workflow da
  comparação em lugar nenhum. É o rascunho anterior ao v2, não uma segunda fonte de verdade.
- **Órfão / risco**: mesmo perfil do item 1 (mesmo diretório ignorado, zero cópia em outro
  lugar).
- **Decisão recomendada**: mesma do v2 se o dono quiser preservar o histórico de iteração; caso
  contrário, é candidato a descarte por ser rascunho superado — decisão do dono, não decidida
  aqui.

### 3. `ComfyUI/user/default/workflows/SMOKE_HunyuanVideo15_W4A4.json`

- **Onde mora hoje**: mesmo diretório, 6 025 bytes, modificado 2026-08-16 10:17.
- **É nosso?** Sim — `grep -l "W4A4"` bate no arquivo (confirmado por conteúdo, não pelo nome).
- **Órfão / risco**: mesmo perfil dos itens 1–2.
- **Decisão recomendada**: mesma recomendação — copiar para caminho rastreado pela bancada.

### 4. Família `*.nunchaku.json` (6 arquivos)

`TXT2IMG-ZIMG.nunchaku.json`, `Zimg-TXT2IMG-_multigpu.app.nunchaku.json`,
`image_qwen_image_edit_2509_relight.nunchaku.json`,
`image_z_image_turbo_fun_union_controlnet.nunchaku.json`,
`templates-image_to_real.nunchaku.json`, `z-image-turbo_00540_.nunchaku.json`.

- **Onde moram hoje**: mesmo diretório `ComfyUI/user/default/workflows/`. Todos modificados
  2026-08-18 11:10–11:26 — a mesma janela em que `CLAUDE.md` registra "Any workflow using a
  Nunchaku SVDQuant loader needs `--disable-dynamic-vram`, verified 2026-08-18 by a real
  generation."
- **São nossos?** Sim, com uma ressalva: `grep -l "Nunchaku\|SVDQuant\|nunchaku"` confirma que
  todos os 6 referenciam nós Nunchaku/SVDQuant. Todos têm uma contraparte-base pré-existente com
  o mesmo nome sem `.nunchaku` e data **anterior** (Maio–Julho). Ou seja: **a variante
  `.nunchaku.json` é nossa** (criada/adaptada por este projeto para testar o loader SVDQuant); **a
  base de onde ela derivou não é** — é biblioteca de workflows do próprio dono, anterior ao
  projeto de quantização.
- **Órfão / risco**: mesmo perfil — `/user/` inteiro ignorado, zero cópia em outro lugar.
- **Decisão recomendada**: copiar as 6 variantes `.nunchaku.json` (não as bases) para um caminho
  rastreado, já que são o artefato que comprova o achado do `--disable-dynamic-vram` registrado
  no `CLAUDE.md`.

### 5. `ComfyUI/custom_nodes/comfy-quant-preflight/` — não é o caso que o ticket temia

O ticket levantou a hipótese "3 arquivos já rastreados, o resto pode não estar". Testado por
execução e o resultado é diferente do suposto:

```
git ls-files | grep -i preflight
  -> custom_nodes/comfy-quant-preflight/__init__.py     (relativo à raiz da bancada,
  -> custom_nodes/comfy-quant-preflight/checks.py           ou seja: F:\COMFY_PORTABLE\custom_nodes\...,
  -> custom_nodes/comfy-quant-preflight/test_checks.py       FORA de ComfyUI\ desde sempre)

find ComfyUI/custom_nodes/comfy-quant-preflight -type f
  -> só __init__.py + __pycache__ (compilado, irrelevante)

stat -c '%s %n' custom_nodes/comfy-quant-preflight/__init__.py \
                ComfyUI/custom_nodes/comfy-quant-preflight/__init__.py   # 2026-08-22
  -> 8945 custom_nodes/comfy-quant-preflight/__init__.py            (o pacote, 207 linhas)
  -> 1471 ComfyUI/custom_nodes/comfy-quant-preflight/__init__.py    (o stub, 34 linhas)
```

Correção de 2026-08-22, medida: esta linha dizia que o arquivo **dentro** do `ComfyUI/` tinha
**8 945 bytes**. Não tem — 8 945 é o tamanho do `__init__.py` **rastreado, fora** do `ComfyUI/`.
O que está dentro tem 1 471 bytes. Os dois números foram trocados, e trocados exatamente na
direção que apaga a conclusão desta seção: o parágrafo abaixo diz que o de dentro é um stub de 34
linhas, e 8 945 bytes em 34 linhas não fecha. É o mesmo modo de falha que já custou uma auditoria
aqui — afirmar a partir do arquivo plausível em vez do arquivo do caminho.

Os três arquivos "rastreados" que o ticket viu no grep de 88 arquivos **já moram fora de
`ComfyUI/`**, em `F:\COMFY_PORTABLE\custom_nodes\comfy-quant-preflight\`, rastreados pela bancada.
O que existe dentro de `ComfyUI\custom_nodes\comfy-quant-preflight\` é um **stub de 34 linhas**
(`__init__.py`, diferente do `__init__.py` de fora — `diff` confirma, são dois arquivos com
conteúdo diferente) cujo próprio docstring explica o desenho:

> "`ComfyUI/` is an upstream git checkout with its own remote, and its .gitignore excludes
> `/custom_nodes/`. Anything kept here is untracked by both repositories [...] So the source of
> truth is `F:/COMFY_PORTABLE/custom_nodes/comfy-quant-preflight/` which this project's repository
> tracks, and this file is the three lines ComfyUI needs to find it."

O stub calcula o caminho da fonte real com `Path(__file__).resolve().parents[3]` — LIDO, não
executado (rodar isso exigiria subir o ComfyUI, proibido nesta rodada): partindo de
`ComfyUI\custom_nodes\comfy-quant-preflight\__init__.py`, `parents[3]` é `F:\COMFY_PORTABLE`, e o
resultado bate com o caminho rastreado acima. A aritmética confere; o carregamento real dentro de
um ComfyUI de pé não foi confirmado nesta sessão.

- **É nosso?** O stub sim (autoria do projeto, decisão deliberada). O pacote real já mora e já é
  rastreado fora de `ComfyUI/` — não é o caso órfão.
- **Órfão / risco**: o pacote real (3 arquivos), não. **O stub de 34 linhas, sim** — ele não
  aparece nem no `git status` do ComfyUI (ignorado por `/custom_nodes/`) nem no `git ls-files` da
  bancada (nunca foi commitado o caminho `ComfyUI/custom_nodes/comfy-quant-preflight/__init__.py`
  em lugar nenhum). Se esse arquivo específico sumir num reinstall, o preflight para de rodar
  **silenciosamente** — o `__init__.py` real de fora tem um `try/except` que garante que uma
  falha de import não derruba o ComfyUI, mas se o *stub* em si desaparecer não há nem esse aviso:
  o diretório simplesmente deixa de existir para o carregador de custom_nodes.
- **Decisão recomendada**: guardar uma cópia do conteúdo do stub dentro do pacote rastreado (por
  exemplo como referência num README do pacote), para que as "três linhas" sejam recriáveis sem
  depender de memória. Não é o pacote que está em risco — é o ponteiro de 34 linhas que o
  ComfyUI usa para achar o pacote.

### 6. `ComfyUI/custom_nodes/comfy_convrot_native/`

- **Onde mora hoje**: `ComfyUI/custom_nodes/comfy_convrot_native/`, 2 arquivos
  (`__init__.py` 4 022 bytes, `compile_support.py` 7 750 bytes), modificados 2026-08-16.
- **É nosso?** Sim, sem ambiguidade — o docstring do `__init__.py` é inequivocamente deste
  projeto: "Make ConvRot W4A4 text encoders execute their native CUDA kernel", cita
  `comfy_kitchen.backends.cuda.convrot_w4a4_linear`, `LTXAVTextEncoderLoader`, e o mesmo número
  medido em Gemma 3 12B (336/336 layers dispatched, 2.354s → 0.539s) que aparece no vocabulário
  de medição deste repo.
- **Órfão dos dois repos**: sim, confirmado por execução —
  `git ls-files | grep -i convrot` na bancada não acha nada sob esse caminho (só
  `tools/convrot_ops_probe.py`, arquivo diferente); `ComfyUI/custom_nodes/` inteiro é ignorado
  pelo checkout upstream. Sem `.git` próprio (diferente do item 7).
- **Risco de perda num update**: mesmo raciocínio da seção geral — o `update_comfyui.bat` sozinho
  não apaga (não toca em `custom_nodes/`), mas não há **nenhuma** cópia rastreada em lugar
  nenhum. É código, não config: implementa um achado central do projeto (o fix de dtype que faz
  o kernel nativo disparar em vez de cair para dequantização), então o risco de perda aqui pesa
  mais que nos workflows JSON.
- **Decisão recomendada**: trazer para dentro de um caminho rastreado da bancada (por exemplo ao
  lado de `tools/`), mesmo padrão de stub-fino usado no item 5 se o dono quiser manter o
  carregamento automático do ComfyUI.

### 7. `ComfyUI/custom_nodes/Comfy-WaveSpeed-Fixed/` — não é órfão, já tem backup próprio

- **Onde mora hoje**: `ComfyUI/custom_nodes/Comfy-WaveSpeed-Fixed/`.
- **É nosso?** É um fork mantido pelo dono, não um pacote de terceiro intocado — mas também não é
  um artefato deste projeto de quantização especificamente (é o node de cache do WaveSpeed).
- **Diferente dos outros itens: tem `.git` próprio.** `git remote -v` dentro dele mostra três
  remotos: `fork -> github.com/JoaoZaokk/Comfy-WaveSpeed-Fixed` (conta do próprio dono — já é
  backup fora da máquina), `origin -> github.com/yannickcruz/Comfy-WaveSpeed-Fixed`,
  `upstream -> github.com/chengzeyi/Comfy-WaveSpeed`. `git status --short` limpo, `git log` mostra
  histórico de commits real (fixes de compatibilidade com ComfyUI 0.8.2/0.3.26, Z-Image Turbo,
  Impact Pack).
- **Órfão? Não.** Por ser um repositório git separado dentro de `ComfyUI/custom_nodes/`, e por
  `custom_nodes/` inteiro ser ignorado pelo checkout upstream, o merge/checkout do
  `update_comfyui.bat` nunca toca nesse diretório de forma alguma — nem por acidente de path.
- **Risco de perda num update**: nenhum risco adicional identificado. O único risco comum a
  qualquer fork é o dono esquecer de dar `git push` pro remoto `fork` depois de commits locais
  novos — não é um risco específico de update do ComfyUI.
- **Decisão recomendada**: nenhuma ação. Já está onde faz sentido, com histórico e remoto
  próprios.

### 8. Candidato verificado e descartado: `ComfyUI/blueprints/VOID_SAM3_Video_Inpaint_DualGPU.json`

O ticket pediu para "confirmar e ampliar" além da lista conhecida. `ComfyUI/blueprints/` tem ~90
arquivos, quase todos datados de Maio/Julho (lote de templates baixados) — um único arquivo,
`VOID_SAM3_Video_Inpaint_DualGPU.json`, tem data de 2026-08-16 10:13, dentro da janela do
projeto, e nome que sugere GPU dupla (o setup deste projeto usa duas GPUs). Testado por conteúdo
antes de incluir na lista:

- `grep` por `convrot`, `W4A4`, `MultiGPU`, `cuda:0`, `cuda:1` dentro do JSON: nenhum marcador do
  projeto de quantização.
- `diff` contra o blueprint padrão `Video Inpaint (VOID).json` (já bundlado pelo ComfyUI): só 4
  linhas de diferença.

**Conclusão: não é artefato deste projeto.** É uma variação pontual do dono num template padrão,
para outra tarefa (inpainting com SAM3), sem relação com ConvRot/W4A4/LTX2.5. Registrado aqui para
que a ausência da lista não pareça um grep que não olhou — olhou, e descartou com evidência de
conteúdo, não só de nome.

---

## Resumo do risco de perda, por item

| Item | Órfão hoje? | `update_comfyui.bat` sozinho apaga? | Sem segunda cópia? |
|---|---|---|---|
| `LTX25-int8-acceptance-v2.json` | sim | não (lido, ver acima) | sim — risco alto em reinstall |
| `LTX25-int8-acceptance.json` (v1) | sim | não | sim — risco alto em reinstall |
| `SMOKE_HunyuanVideo15_W4A4.json` | sim | não | sim — risco alto em reinstall |
| 6× `*.nunchaku.json` | sim | não | sim — risco alto em reinstall |
| `comfy-quant-preflight/` (pacote real, 3 arq.) | **não** | n/a | não — já rastreado fora de `ComfyUI/` |
| `comfy-quant-preflight/__init__.py` (stub, 34 linhas) | sim | não | sim, mas reconstruível a partir do próprio docstring do pacote real |
| `comfy_convrot_native/` (2 arquivos) | sim | não | sim — risco alto, é código de um achado central |
| `Comfy-WaveSpeed-Fixed/` | **não** | n/a | não — tem `.git` e remoto próprios no GitHub |
| `VOID_SAM3_Video_Inpaint_DualGPU.json` | — | — | descartado: não é nosso |

Nenhum item foi movido. Todas as recomendações acima aguardam decisão do dono.
