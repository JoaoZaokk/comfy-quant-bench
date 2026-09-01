# 08 - One converter core: five converters re-implement the same eight-part contract and the parts diverge

Type: grilling
Status: in-progress
Blocked by: 05, 07
Severity: medium
Provenance: TRACED, then EXECUTED 2026-08-22 on the 3090

## Question

Should the five quantizing converters plus `to_native.py` and `svdq_to_bf16.py` route their writing
through one module, with the per-format quantize step as the only plugged-in part?

This is a `grilling` ticket, not a `task`: it is the largest structural change proposed by the audit
and the owner should decide the shape before anyone types it.

## The contract, written out longhand six times

1. refuse source==output, existing output, existing sidecar
2. refuse an already-quantized source (`__metadata__` **and** inline `.comfy_quant` markers)
3. refuse a stale `.partial`
4. plan the output header, computing every `data_offsets` up front
5. stream `("write", tensor)` / `("copy", (start, size))`
6. assert bytes-written == bytes-planned
7. `flush` + `fsync` + `os.replace`, with `finally: partial.unlink()`
8. guard free RAM and free disk before starting

Plan/write/verify/replace lives at `quant_w4a4.py:196-231` + `:269-327`, `quant_w4a8.py:342-393`,
`quant_int8.py:246-289`, `quant_w4a4_smooth.py:263-312`, `quant_mixed.py:601-657`,
`to_native.py:185-227`. Helpers: `read_header` x6, `copy_range` x3 named + 2 inline, `read_tensor` x3
with **two different implementations**, `human_size` x3, `SAFETENSORS_DTYPE` x2 **differing**
(`quant_w4a8.py:211` has `int16`, `quant_mixed.py:65` does not).

## Where duplication has already produced divergence

- **The exclusion net exists in exactly one converter.** `quant_w4a4.py:37-46` defines `EXCLUSIONS`
  and applies it at `:161` and `:170`, with a docstring explaining why the allowlist alone is not
  enough. `quant_w4a8.py:147-161`, `quant_int8.py:75-87` and `quant_mixed.py:164-178` have none -- and
  `quant_int8.py:46-55` imports `PROFILE_PATTERNS` from `quant_w4a8`, so the loosest pattern table in
  the tree is shared by two converters and is the one with no second net. Latent today; live the first
  time anyone writes a looser pattern for Flux / SeedVR2 / Z-Image, all of which CLAUDE.md lists as
  pending.
- **The RAM guard exists in three of five**, and `quant_w4a4` uses a different formula (correct for
  its streaming design). `quant_w4a4_smooth.py` has none, despite `:200-247` accumulating `quantized`,
  `scales` and `new_norms` across every layer before writing -- the exact two-pass shape
  `_ram_guard.py` was written for.
- **The disk guard is three different rules.** `quant_w4a4.py:402-405` checks `estimate + 1 GiB` and
  reports what was free. `quant_w4a8.py:294`, `quant_int8.py:216`, `quant_mixed.py:543` check
  `< source size` and raise a bare `SystemExit` with no numbers. `quant_w4a4_smooth.py` and
  `svdq_to_bf16.py` do not check.
- **The dtype-writing fix landed in one copy.** `quant_w4a8.py:177-208` grew `header_dtype()` and
  `as_bytes()` for ticket 24, with an EXECUTED note at `:194-201` (bfloat16/fp8 `.numpy()` raises).
  `quant_mixed.py` calls the same `ck.quantize_w4a8_int8_weight` at `:565-569`, receives the same
  five-tuple, and writes it at `:618-620` + `:645` handling only `float8_e4m3fn`. A `float8_e5m2`
  scale `KeyError`s; a bfloat16 codebook raises inside `.numpy()` -- **after every layer has been
  quantized.**

## The proposed shape

```python
class Conversion:
    def __init__(self, source, output, *, preflight, sizer): ...
    def plan(self, header, metadata, per_layer): ...
    def commit(self): ...
```

`per_layer(name, tensor) -> list[(key, tensor)]` is the only plugged-in part: two lines for W4A4, six
for W4A8, nine for mixed.

`quant_int8.py:46-55` is the existence proof that the seam works -- it already imports six helpers
from `quant_w4a8`. The move is to finish what it started and invert the direction: the shared code
should not live inside whichever converter was written first.

## Deletion test

Delete `output_header` / `write_streamed_checkpoint` from `quant_w4a4.py` and complexity
**concentrates** -- callers become `plan()` + `commit()`. Delete `_ram_guard.py` and it
**redistributes** into three inline formulas that will drift again, which is what its own docstring
records. The write loops fail the test; `_ram_guard` passes.

## Cheapest immediate fix, if the owner wants to defer the core

`from quant_w4a8 import as_bytes, header_dtype` in `quant_mixed.py`, used at `:620` and `:645`. Three
lines. It does not solve the ticket -- it removes the one divergence that crashes after a full
quantization pass.

## Closing criterion (written before the work)

This ticket closes on a **decision**, not on code: the owner says core / no core / defer, and the
reason is written here. If "core", it spawns implementation tickets and does not itself close until
`grep -c "def write_streamed_checkpoint" tools/*.py` returns 1 and every converter's guard set is the
same set.

## Answer to the one sub-question that needed the card -- EXECUTED 2026-08-22, RTX 3090

**The `quant_mixed.py` dtype gap is latent, not live, on this build.**

`ck.quantize_w4a8_int8_weight(w, group_size=16, convrot_groupsize=256, symmetric=True,
scale_dtype=torch.float8_e4m3fn, codebook=True, codebook_tensor=None, stochastic_rounding=0)`
on a bf16 `[256, 256]` returns:

    [0] torch.int8            (256, 128)
    [1] torch.float8_e4m3fn   (256, 16)
    [2] torch.float32         (256,)
    [3] NoneType              -
    [4] torch.float32         (16,)

Nothing here is `bfloat16` or `float8_e5m2`, so `quant_mixed.py:618` does not `KeyError` and
`:645`'s `.numpy()` does not raise. **The crash-after-a-full-quantization-pass scenario cannot
happen with these arguments today.**

What that does and does not mean:

- It **lowers the urgency** of the "cheapest immediate fix" (`from quant_w4a8 import as_bytes,
  header_dtype`). Do it anyway -- it is three lines -- but it is not on fire.
- It **does not weaken the ticket.** The reason `quant_w4a8.py` grew `header_dtype()`/`as_bytes()`
  at `:177-208` in the first place was a real failure with an EXECUTED note attached, under
  different arguments. `quant_mixed.py` calls the same op and would not survive them. The divergence
  is real; it is the *trigger* that is currently absent.
- It is exactly the shape of thing a converter core makes impossible to have: one writer, one dtype
  table, and the question never arises per-file.

**Caveat that travels with this:** one call, one argument set, one build. `scale_dtype` is a
parameter -- pass `float8_e5m2` and the answer may change. Not tested.

---

## DECISAO DO DONO, 2026-08-31: NUCLEO

Perguntado de novo depois de o inventario ser refeito, ele decidiu pelo nucleo, e acrescentou uma
peca que este ticket nao tinha: a **superficie**. Palavras dele -- "juntar esses codigos em Py e o
`quant_w4a4` ser uma funcao, um def, que vai ser chamado quando alguem digitar `--w4a4`".

Ele tambem registrou por que tinha adiado antes: *"na hora eu queria um resultado rapido"*. Nao foi
recusa; foi prioridade.

### O que foi feito

`tools/_conversion.py` -- o contrato de oito partes escrito uma vez. `Conversion(source, output)`
com `.refuse_unsafe()`, `.guard()` e `.commit(entries, metadata)`. A costura de plano aceita tres
formas de payload -- `("copy", (inicio, tamanho))`, `("write", tensor)` e
`("write", callable)` -- e a terceira e o que faz o nucleo servir tanto quem transmite
(`quant_w4a4`, que quantiza dentro do laco de escrita) quanto quem acumula (`quant_w4a8`,
`quant_mixed`), sem obrigar o primeiro a segurar o modelo inteiro em RAM so para caber na costura.
Essa era a objecao real contra a forma que este ticket desenhou.

`tools/test_conversion_core.py` -- **32 checagens, 0 falhas, EXECUTADO** contra safetensors de
verdade em disco temporario. Sem mock: o ponto de um contrato de escrita e o byte no disco.

`tools/convert.py` -- a porta unica, por SUBCOMANDO e nao por flag.

### Por que subcomando e nao `--w4a4`

Nao e preferencia. EXECUTADO: os namespaces de `--profile` sao disjuntos entre as ferramentas --
`mixed --profile gemma` e invalid choice, `w4a8 --profile zimage` e invalid choice. Um argparse
plano nao expressa "esta flag aceita estes valores QUANDO aquela outra esta presente"; viraria
validacao pos-parse, o mesmo defeito com outro nome. Some-se a isso quatro grafias do mesmo
conceito na arvore (`--profile`, `--arch`, `--auto-detect`, e o `--profile` sem `auto` do mixed) e
um `--output` obrigatorio em dois dos sete e opcional em cinco.

### Por que os sete arquivos continuam existindo

EXECUTADO por um refutador que tentou derrubar "da para fundir sem quebrar nada", e derrubou: oito
arquivos importam esses modulos pelo nome e quebram se sumirem, e `test_svdq_write_contract.py:229`
lista os SETE nomes de arquivo literalmente. A porta da frente e nova; os modulos seguem sendo os
modulos.

### Divergencias que o nucleo resolve, e a regra que adotou

| parte | como estava | o que vale agora |
|---|---|---|
| disco | 3 regras; 3 dos 7 sem nenhuma; duas levantavam sem numero | `estimativa + 1 GiB`, dizendo livre e quanto falta |
| RAM | 4 de 7; o pior caso (`smooth`, que acumula tudo) sem guarda | obrigatoria para quem declara acumulo |
| bytes conferidos | `smooth` levantava `RuntimeError("length mismatch")` sem numero | a mensagem carrega escrito, planejado e a diferenca |
| `SAFETENSORS_DTYPE` | duas tabelas; `mixed` sem `int16` | uma tabela |
| fp8 / bf16 na escrita | `header_dtype`/`as_bytes` so no `w4a8`; `e5m2` dava KeyError **depois de quantizar tudo** | unico caminho de escrita |
| recusa em dry-run | `mixed` condicionava as recusas a `not args.dry_run` | incondicional: um ensaio que pula a checagem do caminho real nao e ensaio |

### O que este ticket NAO fecha ainda

O criterio de fechamento foi escrito antes de alguem olhar o resultado, e ele exige mais do que
existe hoje:

    grep -c "def write_streamed_checkpoint" tools/*.py devolve 1
    e o conjunto de guardas de todo conversor e o mesmo conjunto

O nucleo existe e esta testado, e a porta unica existe. **Os sete ainda nao foram migrados para
ele.** Enquanto nao forem, o contrato continua escrito oito vezes -- sete nos conversores e uma no
nucleo -- e isso e pior do que sete, nao melhor, porque agora ha uma copia que se parece com a
fonte da verdade sem ser. Migrar e o trabalho seguinte, e este ticket segue aberto ate la.

---

## Divergencia nova, achada em 2026-08-31: duas `PROFILE_PATTERNS` com as MESMAS chaves e significados incompativeis

Ao varrer o disco atras de modelos com perfil conhecido, este comando devolveu **zero candidatos**:

```python
todos = {}
for d in (quant_w4a4.PROFILE_PATTERNS, quant_w4a8.PROFILE_PATTERNS,
          calibrate_activations.PROFILE_PATTERNS):
    todos.update(d)
```

E devolveu zero porque as tabelas colidem. Mesma chave, alvos diferentes:

```
quant_w4a4  hunyuan_video_15  ->  ^double_blocks\.\d+\....\.weight$    NOME DE TENSOR
calibrate   hunyuan_video_15  ->  ^double_blocks\.\d+\.(?:img|txt)_attn\.(?:qkv|proj)...$   CAMINHO DE MODULO
```

A primeira casa chaves do cabecalho safetensors. A segunda casa a arvore de modulos **depois** que o
ComfyUI renomeia na carga (`img_attn_qkv` vira `img_attn.qkv`, e sem `.weight`). Sao para coisas
diferentes e as duas estao certas no proprio contexto -- o problema e que **compartilham o nome**.

O `update()` fez a segunda ganhar, e a varredura passou a testar padroes de modulo contra chaves de
arquivo. Nada casou, e **nada avisou**: nao houve erro, nao houve aviso, so um zero plausivel.
Custou tres rodadas de depuracao, e a primeira delas tinha `except Exception: pass`, que engoliu a
evidencia -- o defeito que este repo cataloga como "ausencia num grep nunca e ausencia no sistema".

Isto e a mesma classe de divergencia que este ticket ja lista (`read_tensor` com duas
implementacoes, `SAFETENSORS_DTYPE` com duas tabelas), com um agravante: as outras divergem em
comportamento e esta diverge em SIGNIFICADO. Um `from X import PROFILE_PATTERNS` pega a tabela
errada sem nenhum sinal.

Conserto quando o nucleo for adiante: nomes distintos (`PADROES_TENSOR` e `PADROES_MODULO`) ou um so
lugar que exponha os dois explicitamente. Nao renomear pela metade -- duas tabelas com nomes
parecidos e pior que duas com o mesmo nome.

## Estado da ADOCAO, medido em 2026-09-01

A decisao esta tomada e o nucleo esta escrito e testado. **O que falta e ninguem ter trocado para
ele**, e o ticket nao dizia isso em lugar nenhum.

```
tools/_conversion.py              existe, 8 partes do contrato
tools/test_conversion_core.py     32 checagens, 0 falhas, EXECUTADO
importadores em producao          ZERO
```

O unico modulo em toda a arvore que faz `import _conversion` e o proprio teste dele. Medido com
`rg "_conversion" -g "*.py"`: tres ocorrencias, duas em docstring e uma no teste.

E os cinco conversores continuam cada um com o seu:

```
quant_w4a4.py         3 ocorrencias de (.partial|os.replace|fsync)
quant_w4a4_smooth.py  3
quant_int8.py         3
quant_mixed.py        3
to_native.py          3
```

**O docstring do `convert.py` dizia que o contrato "virou um so"**, o que le como trabalho feito.
Corrigido no mesmo dia, junto com a afirmacao gemea no `CLAUDE.md`. Uma implementacao unica foi
escrita e testada; a troca nao aconteceu.

### O que isto muda no criterio de fechamento

Nada -- o criterio ja exigia que todo conversor passasse pelo nucleo com o mesmo conjunto de
guardas, e ele continua nao satisfeito. O registro existe para que ninguem leia
"`_conversion.py` existe e passa 32 testes" como "os conversores usam".

### Ordem sugerida para a adocao, e por que ela comeca no `to_native`

`to_native.py` **nao quantiza** -- e um remapeamento de nomes -- entao nao passa pelo preflight
nativo e **roda inteiro sem GPU**. Ele tambem tem uma saida ja existente no disco
(`beyond-reality-zimage-v2_native.safetensors`) para comparar **byte a byte** contra o resultado da
versao migrada. E o unico dos seis onde a migracao pode ser verificada de ponta a ponta sem placa,
o que faz dele o piloto certo. Os quatro quantizadores precisam de uma janela de GPU para uma
conversao real depois de migrados.

Nao coberto por esta nota: nada foi migrado ainda, so contado.

## O link do dono: RECEBIDO e RESPONDIDO em 2026-08-30. Nao perguntar de novo.

Em 2026-08-22 o dono disse *"eu acho que te mandei um github sobre isso no passado"*. Ele mandou em
**2026-08-30T08:13Z**, domingo de manha, ja com a ressalva de que nao tinha certeza:

    https://github.com/RealJonathanYip/ComfyUI-QuantFunc
    "nao sei, honestamente, mas e uma das abas que deixei abertas para ver depois"

Respondido as 08:17 do mesmo dia: **nao e o ticket 08.** E engine de inferencia C++/CUDA em processo
worker (`quantfunc.dll`), que quantiza **em runtime**, "zero Python model dependencies". Este ticket
e sobre cinco **escritores de checkpoint** que divergiram num contrato de oito partes; o QuantFunc
nem escreve safetensors com contrato. Coisas diferentes.

O que provavelmente deixou a aba aberta: SVDQ offline + NVFP4, 2x-11x. **NVFP4 e Blackwell e esta
bancada e Ampere sm_86.** Vale como referencia de arquitetura, nao como base deste ticket.

**Proveniencia:** veredito por **LEITURA do README**, nao por execucao. A diferenca aqui e
categorica (engine de inferencia contra escritor de checkpoint), entao o README basta para
sustenta-la — mas o rotulo fica, porque nesta bancada trace e medicao nao se parecem no texto.

**Consequencia para o ticket:** nao ha terceiro desenho esperando. O `_conversion.py` que ja existe
**e** a forma, e o que falta e so a adocao. A pergunta que bloqueava esta fechada.

### Como esta pergunta foi feita duas vezes, que e a licao

A memoria do projeto registrou "pergunta pendente ao dono" as **07:52:41Z** de 2026-08-30. Ele
respondeu **21 minutos depois**. A memoria nunca foi atualizada, e em 2026-09-01 ela foi carregada
e a pergunta refeita ao dono — que ja tinha respondido. Custou o tempo dele.

O cortex nao ajudou e nao ajudaria: `cortex serve` esta fora do ar, a busca cai para FTS5 keyword,
e ela devolveu **zero** hits para este assunto. O que achou foi um grep direto nos transcritos
locais (`~/.claude/projects/<slug>/*.jsonl`), que sao arquivos comuns. **Ausencia no cortex nao e
ausencia no historico** — o mesmo defeito que este repo cataloga em
`arquivo-plausivel-nao-e-o-caminho`.

## ADOCAO FEITA, 2026-09-01. O ticket NAO fecha ainda, e o motivo esta escrito aqui.

Os **sete** escritores passaram a usar `tools/_conversion.py`. Auditado por contagem, nao por
memoria:

```
conversor            importa nucleo  commit  guard  refuse_unsafe  .partial proprio
quant_w4a4                 sim          1      1         1              0
quant_w4a8                 sim          1      1         1              0
quant_int8                 sim          1      1         1              0
quant_w4a4_smooth          sim          1      1         1              0
quant_mixed                sim          1      1         2              0
to_native                  sim          1      1         1              0
svdq_to_bf16               sim          1      1         1              2  (ver pendencia)
```

### O criterio de fechamento, item a item

O criterio dizia: fecha com uma decisao escrita, **e** so quando houver um unico escritor e todo
conversor tiver o mesmo conjunto de guardas.

- **Decisao**: tomada em 2026-08-31, NUCLEO. Feito.
- **Um escritor so**: `Conversion.commit` e o unico no caminho de producao. Feito, com uma
  pendencia nomeada abaixo.
- **Mesmo conjunto de guardas**: todos chamam `conv.guard(disco, accumulated=...)`. Feito -- e
  isto **corrigiu divergencias reais**, nao so mudou de lugar:
  - `quant_w4a4_smooth` era o unico dos sete **sem guarda de RAM e sem guarda de disco**, e e o
    que mais acumula (pesos, escalas E normas reescritas). Ganhou as duas.
  - `svdq_to_bf16` era o unico **sem recusa nenhuma de saida existente**. Ganhou, com
    `allow_quantized_source=True`, porque a entrada dele e quantizada por construcao.
  - `quant_mixed` era o unico que condicionava a recusa de saida existente a `not --dry-run`, o
    que fazia o ensaio seco pular a checagem que a execucao real faria. Agora e incondicional.

**Mesmo assim nao fecho**, porque nenhum byte de DADO quantizado foi conferido: o kernel nao rodou
em verificacao nenhuma. Fechar aqui seria fechar sobre codigo lido, e o criterio deste repo e o
oposto disso.

### O que ja esta provado sem GPU

| conversor | como | resultado |
|---|---|---|
| `to_native` | reconverteu e comparou o arquivo inteiro | **sha256 identico, 11,46 GiB** |
| `quant_w4a4` | header reconstruido sem kernel, comparado byte a byte | **identico, 238 264 bytes, 432 camadas** |
| todos | 9 suites | passam |

A segunda linha e `tools/verificar_migracao.py`, e ela existe porque `plan_lazy` carrega dtype,
forma e nbytes explicitamente -- o plano inteiro sai sem chamar kernel. Os quatro que ACUMULAM nao
tem esse verificador: as formas de saida deles (`s_rel`, `s_channel`, `codebook`) saem do kernel.

### O que falta, com criterio escrito antes

`bench/janela_gpu_migracao.md`, escrito antes de qualquer corrida com placa: os pares a
reconverter, os parametros que precisam sair do `.quant.json` de cada saida, e a regra de que
**qualquer diferenca reprova** -- inclusive diferenca so de header, ja que a convencao foi
alinhada de proposito para permitir comparacao byte a byte.

### Duas coisas que a migracao expos, e que ler o codigo nao teria exposto

**O nucleo tinha um buraco.** `to_native` funde `to_{q,k,v}` num `qkv`, entao um destino nasce de
varias faixas da fonte, e o nucleo so tinha `plan_copy` de faixa unica. Dai `plan_copy_many`.

**O nucleo era MAIS FRACO que o que substitui.** `commit()` abria o `.partial` com `"wb"`, que
trunca, onde os seis escritores usam `"xb"`. Pego pelo `test_svdq_write_contract` durante a
propria migracao -- que e o risco real de extrair um contrato: a extracao perde uma garantia e
ninguem nota. E aquele teste exigia o literal `"xb"` dentro de cada um dos sete arquivos, o que
teria bloqueado exatamente a migracao que este ticket existe para fazer; passou a afirmar a
garantia (o escritor carrega, ou delega a um nucleo que carrega) e cobra o literal do nucleo.

**E a convencao do header teve que ser decidida por contagem, nao por gosto.** Cinco de cinco
quantizadores escrevem `__metadata__` PRIMEIRO com `ensure_ascii=False`; o nucleo fazia o
contrario nos dois. Adotar o nucleo teria mudado em silencio o layout de todo checkpoint
reconvertido -- e destruido a unica verificacao forte que a migracao tem. O nucleo passou a seguir
a convencao existente.

### Pendencia nomeada

`svdq_to_bf16.write_checkpoint()` saiu do caminho de producao mas continua no arquivo: o
`test_svdq_write_contract.py` a exercita direto em tres cenarios, e apagar a funcao apagaria o
teste junto. Redirecionar aquele teste para `Conversion.commit` e entao remover a funcao. Os tres
cenarios ja tem cobertura equivalente em `test_conversion_core.py` (partes 3b, 5, 6, 7), entao o
redirecionamento e para preservar as anotacoes de proveniencia daquele arquivo.

**Essa pendencia FECHOU** em `f80f3de`: o teste foi redirecionado e depois a funcao apagada, 87
linhas, junto com `COPY_CHUNK` e `DTYPE_NAMES`, que so ela usava.

---

## A JANELA DE GPU RODOU, 2026-09-01. Quatro pares byte-identicos.

Executado na 3090 com `CUDA_VISIBLE_DEVICES=0` e `Assert-GpuLock` a mao (conversor nao passa por
`_timing.compare()`, entao nao toma o lock sozinho). O criterio estava escrito antes, em
`bench/janela_gpu_migracao.md`: **qualquer diferenca reprova, inclusive so de header.**

| conversor | fonte | saida | sha256 |
|---|---|---|---|
| `quant_w4a4` | hunyuanvideo1.5 fp16 | 8 507 690 240 B | `A3485DAA…A4732FBF` **identico** |
| `quant_w4a8` | hunyuanvideo1.5 fp16 | 8 847 567 376 B | `3ED43444…A41E76D7` **identico** |
| `quant_mixed` | wan2.1 vace 1.3B | 2 310 437 144 B | `212B9111…0B40A142` **identico** |
| `quant_mixed` | beyond-reality-zimage-v2 | 3 403 133 032 B | `4463AC4E…2113CF5C` **identico** |

O quarto par nao estava no plano e foi acrescentado de proposito: sem ele o `quant_mixed` estaria
provado numa arquitetura so. Wan (300 camadas, 2 w4a4 / 298 w4a8) e Z-Image (170 camadas, 117 /
53) exercitam ramos de selecao bem diferentes do mesmo codigo.

### Os tres sem par, e a aceitacao mais fraca que isso obriga

| conversor | o que rodou | resultado |
|---|---|---|
| `quant_int8` | conversao real + carga pelo loader normal + contagem de despacho | 432 modulos `int8_tensorwise`, **8/8 forwards quantizados, 0 dequantize**, `int8_linear=comfy_kitchen.backends.cuda` |
| `svdq_to_bf16` | recuperou 11,46 GiB do `svdq-int4_r32-z-image-turbo` + carga pelo loader normal | carregou como `Lumina2`, 6 154 908 736 params, bf16, 34 chaves qkv fundidas |
| `quant_w4a4_smooth` | **nao rodou** | ver abaixo |

Isso e "carrega e despacha", **nao** e byte a byte, e a diferenca importa: nada nas duas linhas
acima compara contra uma saida anterior, porque nao existe uma.

Nota util sobre o `svdq_to_bf16`: o docstring dele avisa que manter o layout FUSIONADO significa
que "qualquer loader le" e falso. Para **esta** arquitetura o aviso nao morde -- o `Lumina2` do
ComfyUI usa `attention.qkv`, entao o fundido e exatamente o que ele quer. O aviso continua valendo
para loaders que esperam `to_q`/`to_k`/`to_v` separados.

### O `smooth` nao rodou, e o motivo nao e o codigo

As DUAS entradas que ele exige nao estao nesta maquina. Medido com busca recursiva `-Force` nos
dois roots (`ComfyUI/models` e o share `D:`), que enxerga `.disabled` e ocultos:

- nao ha `gemma_3_12B_it_heretic.safetensors` (o BF16 fonte) -- so fp8, w4a8, LoRAs e GGUF;
- nao ha **nenhum** checkpoint Gemma no formato `convrot_w4a4` para `--calibrate-with`.

Duas saidas foram tentadas e as duas fecharam por motivo medido, nao por desistencia:

- **fonte fp8 + calibragem no gemeo w4a8**: a calibragem rodou INTEIRA (96 normas, 6 prompts,
  ~3 min de 3090) e so entao morreu com `KeyError: 'F8_E4M3'` dentro de `quant_w4a8.read_tensor`;
- **Gemma-3 1B, que e fonte valida** (BF16 puro, 26 camadas, nomes exatos), **nao serve de
  calibragem**: `comfy.sd.load_clip` o detecta como `lumina2`/`gemma3_4b` em vez de LTXV e o
  tokenizer levanta `ValueError: invalid tokenizer`.

Entao **`conv.guard()` e `conv.commit()` do `smooth` continuam sem ter rodado depois da
migracao**, e isso esta impresso no fim de `tools/test_smooth_guards.py` toda execucao, para que
7/7 OK nunca leia como "o smooth foi verificado".

### O caso de CONTROLE achou dois defeitos reais no `smooth`

O teste de recusas foi escrito com um controle negativo -- argumentos validos tem de PASSAR da
guarda -- porque sem ele um conversor que morresse em toda invocacao passaria em todas as
recusas. **O controle falhou, e a falha era o achado.**

- **Guarda vazia ausente.** Apontado para um Wan 2.1, o `smooth` imprimia `Layers: 0 quantized: 0`
  e saia com **rc=0**. `LAYER_RE` nao casa nada, `selected` e `norm_keys` saem vazias, e a
  checagem de `missing` compara duas listas vazias e aprova. Um `--dry-run` -- que e exatamente o
  que se roda ANTES de gastar horas -- respondia SUCESSO para uma conversao sem nada a converter.
  Os outros quatro ja recusavam (`quant_w4a4.py:386`, `quant_w4a8.py:248`, `quant_int8.py:139`,
  `quant_mixed.py:581`); so este nao.
- **Guarda de dtype ausente.** Validava NOMES e nunca o dtype, e `load()` le por `read_tensor`,
  cujo `TORCH_DTYPES` so tem BF16/F16/F32. Custo do erro tardio: ~3 min de GPU jogados fora e um
  rastro apontando para outro arquivo. Depois do conserto **recusa em 1,8 s**, e o teste cobra
  esse tempo (limite 60 s) justamente para pegar a regressao de mover a checagem para depois da
  calibragem.

`tools/test_smooth_guards.py`, novo: **7/7, 0 falhas**, seis recusas mais o controle.

### Coisa que so a corrida ensina

**O `convrot_groupsize` tem de ser potencia de 4, nao de 2.** Tentar 128 no Gemma 1B (cujo K=1152
e divisivel por 128) levanta `ValueError: Regular Hadamard size must be a power of 4, got 128` em
`comfy_kitchen/tensor/int8_utils.py:22`. Explica por que 64 e 256 sao os unicos valores usados
nesta arvore. **O preflight de backend pegou isso antes de qualquer trabalho** -- que e a coisa
mais util que ele fez o dia inteiro.

### Um par que NAO da para reconverter, por decisao e nao por defeito

`zimage-v2-mixed`, `zimage-v2-w4a4` e os `zimage-v2-mixed-t0.*` foram construidos com analises
anteriores a 2026-08-22, que nao carregam `source_identity_sha256`. O `quant_mixed` **recusa** uma
analise sem chave de proveniencia em vez de pular a checagem -- que era exatamente como a guarda
antiga passava. Reproduzi-los exige recalibrar. A guarda esta certa; o registro existe para que
ninguem leia "nao reconverti" como "reconverti e deu diferente".

### Estado do criterio de fechamento

    grep -c "def write_streamed_checkpoint" tools/*.py devolve 1     -> FEITO (e zero: a funcao
                                                                        foi apagada, o nucleo e o
                                                                        unico escritor)
    o conjunto de guardas de todo conversor e o mesmo conjunto       -> FEITO
    nenhum byte de dado quantizado conferido                         -> RESOLVIDO: quatro pares
                                                                        byte-identicos, dois
                                                                        carregam e despacham

**Falta so o `smooth` escrever um byte pelo caminho novo**, e isso depende de um download de
~22 GiB (o Gemma 3 12B BF16) mais uma conversao W4A4 dele para servir de calibragem. E decisao do
dono: rede, disco e tempo. Sem isso, o ticket fecha com uma linha explicita dizendo que seis dos
sete foram verificados executando e o setimo so nas recusas.
