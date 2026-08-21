# Não existe modelo do formato API, e três achados são sintoma disso

Type: task
Status: resolved

## Question

`tools/comfy_run_workflow.py` converte um workflow de formato UI para o formato API
do ComfyUI e o submete. O formato API tem um tipo central — o valor de um input é
**ou** um literal **ou** um fio `[origin_id, slot]` — e esse tipo não existe em lugar
nenhum do código. Tudo mexe em `dict` cru.

Consequências espalhadas pelo arquivo:

- O override de seed (`main()`, linhas 257-267) pergunta "isto é um fio?" com
  `not isinstance(node["inputs"][key], list)` — invariante central expressa como
  teste de tipo sobre dict sem tipo, três níveis dentro do orquestrador.
- `ui_to_api` devolve avisos como **strings**, e o chamador não distingue grave de
  informativo sem parsear inglês (ver `03`).
- `main()` mexe no prompt convertido por chave de string depois da conversão.

O judo: modelar o que já existe.

```python
Wire  = tuple[str, int]                 # ["origin_id", slot]
Value = str | int | float | bool

@dataclass
class Node:
    class_type: str
    inputs: dict[str, Value | Wire]
    def set_widget(self, key: str, v: Value) -> bool   # no-op quando é Wire
```

O `isinstance` some porque "é fio?" vira pergunta que o modelo responde.

## Ordem

Este ticket vem **antes** de `03`, e a versão anterior deste mapa tinha a ordem
invertida. A revisão termonuclear concluiu que o modelo dissolve os sintomas; pôr o
sintoma antes seria consertar duas vezes. `04` é exceção e não depende daqui.

## Critério de fechamento

Fecha quando `Wire`, `Value` e `Node` existirem como tipos, o override de seed for um
método do modelo em vez de mutação de dict no `main()`, e nenhum
`isinstance(..., list)` sobreviver como teste de "é fio".

Pode fechar **sem** o `main()` estar decomposto: isso é o ticket `06`.

## Resolução — 2026-08-21, commit `3d63ee1`

Feito por agente sonnet interrompido antes de reportar. **Verificado pelo orquestrador,
por execução**, não pelo relatório dele (não houve relatório).

A suíte que o agente escreveu compara contra goldens que ela mesma gera — circular. O
círculo foi quebrado importando a versão do `git HEAD` e a do disco no mesmo processo e
passando a mesma entrada nas duas:

```
PASS  prompt convertido identico  (18 nos antes, 18 depois)
PASS  avisos identicos  (1 antes, 1 depois)
```

Critério, item por item: `Wire`/`Value`/`Node` existem (linhas 51, 52, 56); o override de
seed é `node.set_widget(key, args.seed)` na 320, método do modelo; a linha 261 do HEAD
(`not isinstance(node["inputs"][key], list)`) sumiu. Os `isinstance` restantes são sobre
forma de JSON cru — spec de widget, tupla de link — não sobre fio.

`py_compile` limpo. Os 12 testes do agente passam, saída 0.

**Não coberto:** `main()` inteiro — POST, poll, relatório, heurística de cache — intocado
e não verificado. Nenhuma GPU foi usada; nada foi renderizado. O fixture
`object_info_ltx25.json` cobre 16 classes do ComfyUI 0.33.0 e envelhece em silêncio se as
`INPUT_TYPES` mudarem upstream; a suíte não detecta isso sem servidor vivo.
