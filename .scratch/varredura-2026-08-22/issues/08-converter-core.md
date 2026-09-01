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
