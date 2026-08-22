# Artefatos nossos que moram dentro de `ComfyUI/`

Type: task
Status: resolved

## Question

A linha de escopo deste mapa exclui `ComfyUI/` — checkout git separado, código
upstream de terceiro. Mas há coisa **nossa** morando lá dentro, e ela caiu fora do
escopo por acidente de diretório, não por decisão.

Conhecido:

- `ComfyUI/user/default/workflows/LTX25-int8-acceptance-v2.json` — autorado por nós.
  É o lado ComfyUI da comparação end-to-end inteira (512x512, 49 frames, 3 passos,
  cfg 1.0, euler, seed 1234, 25 fps). **Sem ele a comparação não se reproduz.**
- Outros arquivos no mesmo diretório que podem ser nossos: `LTX25-int8-acceptance.json`
  (v1), `SMOKE_HunyuanVideo15_W4A4.json`, os `*.nunchaku.json`.
- `custom_nodes/comfy-quant-preflight/` — pacote nosso. Três arquivos dele já estão
  entre os 88 rastreados, o que sugere que o resto não está.

Levantar o que é nosso lá dentro e decidir se fica onde está — com o risco de sumir
num update do ComfyUI — ou vem para o repo da bancada.

## Como este ticket apareceu

Não estava no desenho inicial. Surgiu de uma pergunta do dono do repo: "o que
exatamente é tirar o ComfyUI, e o que envolve ComfyUI que mexemos e você colocou no
plano com outro nome?". A resposta expôs que a linha de escopo separava por
diretório, e o diretório é de terceiro enquanto o arquivo é nosso.

## Critério de fechamento

Fecha quando existir a lista do que é nosso dentro de `ComfyUI/` e, para cada item, a
decisão de onde ele mora, com o motivo.

Pode fechar **sem** mover arquivo: decidir que ficam onde estão é resposta válida,
desde que o risco de perda num update esteja escrito.

## Resolução

Levantamento completo em
[`../artefatos-em-comfyui.md`](../artefatos-em-comfyui.md). Nenhum arquivo foi movido — a decisão
de mover é do dono, conforme o critério permite.

**O que "resolved" aqui não quer dizer, escrito em 2026-08-22 porque um leitor só deste ticket não
tinha como saber.** `resolved` significa que a lista existe e cada item tem decisão escrita. **Não**
significa que os órfãos deixaram de ser órfãos. O de maior risco continua exatamente onde estava:

`ComfyUI/custom_nodes/comfy_convrot_native/` — 2 arquivos (`__init__.py`, `compile_support.py`),
o código do nó **ConvRot W4A4 Native (Text Encoder)** de `W4A4_PROGRESS.md:202`. Conferido por
execução em 2026-08-22, os dois lados:

```bash
git ls-files | grep -i convrot_native                 # nada: fora do repo da bancada
git -C ComfyUI ls-files | grep -i convrot_native      # nada: fora do checkout upstream
ls -a ComfyUI/custom_nodes/comfy_convrot_native       # sem .git próprio
```

Órfão dos três. `../artefatos-em-comfyui.md` já o classifica como "risco alto, é código de um
achado central" — e o critério de fechamento deste ticket permite fechar assim, desde que o risco
de perda esteja escrito, que está. Mas fechar com o risco aceito não é o mesmo que fechar com o
risco eliminado, e essa distinção precisava estar no ticket, não só no levantamento.

O que mudou: nada em `ComfyUI/`. Foi criado o arquivo de levantamento acima.

Comando que provou cada afirmação central (todos executados, não só lidos):

- `git -C F:\COMFY_PORTABLE ls-files | grep "^ComfyUI/"` → zero linhas: a bancada não rastreia
  nada dentro de `ComfyUI/`.
- `grep "/user/\|/custom_nodes/" ComfyUI/.gitignore` → ambos ignorados por inteiro no checkout
  upstream, confirmando que qualquer arquivo nosso salvo ali é órfão dos dois repositórios.
- `python_embeded\python.exe -s -c "..."` sobre cada `.json` suspeito, testando por conteúdo
  (`seed 1234`, `euler`, `W4A4`, `Nunchaku`) em vez de confiar no nome do arquivo.
- `diff` entre `custom_nodes/comfy-quant-preflight/__init__.py` (rastreado, fora de `ComfyUI/`) e
  `ComfyUI/custom_nodes/comfy-quant-preflight/__init__.py` (dentro): são arquivos diferentes — um
  é o pacote, o outro é um stub de 34 linhas que importa o pacote de fora. Isso derrubou a
  hipótese do próprio ticket ("3 de X arquivos rastreados, resto pode não estar") — o pacote real
  já está inteiro fora de `ComfyUI/` e rastreado; só o stub é órfão.
- `git -C ComfyUI/custom_nodes/Comfy-WaveSpeed-Fixed remote -v` e `log --oneline` → tem `.git`
  próprio com 3 remotos, incluindo um sob a conta do próprio dono no GitHub. Não é órfão.
- Leitura (não execução — proibido subir GPU/servidor nesta rodada) de `update/update.py`: o
  `repo.stash()` usa `include_untracked=False`/`include_ignored=False` por padrão, e o
  checkout/merge só escreve paths que existem na árvore git remota. Rodar
  `update_comfyui.bat` sozinho não apaga os órfãos listados; o risco real é qualquer reinstalação
  mais bruta (`git clean -dfx`, reclone, restauração de backup) que não preserve `user/` ou
  `custom_nodes/`.

O que ficou sem cobertura: não executei `update_comfyui.bat` de verdade (proibido: mexeria no
checkout do ComfyUI e não foi pedido), então "não apaga" vem de ler `update.py`, não de rodar —
isso está marcado explicitamente no arquivo de levantamento e aqui. Não abri o
`ComfyUI/blueprints/` inteiro arquivo por arquivo — só o único candidato datado dentro da janela
do projeto foi conferido por conteúdo; os ~90 arquivos restantes têm data de Maio/Julho,
anterior ao projeto, e não foram individualmente checados. Não conferi
`ComfyUI/user_BACKUP_20260531-152039/` nem `ComfyUI/models_BACKUP_20260531-142811/` — datados de
antes da janela do projeto (31/05), fora do escopo de "artefato deste projeto".
