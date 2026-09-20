# Janela de GPU: verificar a migracao dos conversores para o nucleo

Escrito em 2026-09-01, **antes** de qualquer corrida com placa, para que o criterio nao seja
inventado depois de ver o resultado. Os sete escritores desta bancada passaram a usar
`tools/_conversion.py`; o que falta e provar que os **bytes de dado** nao mudaram.

## O que ja esta provado, sem GPU

| conversor | verificacao | resultado |
|---|---|---|
| `to_native.py` | reconverteu e comparou o ARQUIVO inteiro | **sha256 identico em 11,46 GiB** |
| `quant_w4a4.py` | header reconstruido sem kernel e comparado byte a byte | **identico, 238 264 bytes, 432 camadas** |
| todos | 9 suites de teste | todas passam |

`tools/verificar_migracao.py` re-roda a segunda linha a qualquer momento, sem placa.

## O que NAO esta provado e precisa da placa

**Nenhum byte de dado quantizado foi conferido.** O kernel nao rodou em nenhuma verificacao acima.
Um conversor que planeje o header certo e escreva peso errado passa em tudo que ja foi feito.

E os quatro que ACUMULAM (`quant_w4a8`, `quant_int8`, `quant_mixed`, `quant_w4a4_smooth`) nem o
header tem verificado, porque as formas de saida deles (`s_rel`, `s_channel`, `codebook`) saem do
kernel e nao dao para reconstruir sem ele.

## Criterio de aceitacao, escrito antes

Para cada par abaixo: reconverter com o codigo migrado, para um caminho NOVO, e comparar com o
arquivo que ja esta no disco.

**Passa** quando o `sha256` do arquivo inteiro bate. **Reprova** com qualquer diferenca, e a
diferenca vira achado -- nao se ajusta o criterio.

Uma diferenca de header com dado identico tambem **reprova**: a convencao de header foi alinhada
de proposito (`__metadata__` primeiro, `ensure_ascii=False`) justamente para que a comparacao
pudesse ser byte a byte. Se o header divergir, alguma escolha dessas se perdeu.

## Os comandos, na ordem

Tomar o lock a mao **nao** e necessario para os benchmarks, mas E necessario para conversores --
eles nao passam por `_timing.compare()`. Ver CLAUDE.md, "A janela da GPU".

```bash
.\python_embeded\python.exe -s .\tools\verificar_migracao.py
```

```bash
.\python_embeded\python.exe -s .\tools\quant_w4a4.py --input ComfyUI\models\diffusion_models\hunyuanvideo1.5_720p_t2v_fp16.safetensors --output F:\_migracao\hv15_w4a4.safetensors --convrot-groupsize 256
```

```bash
.\python_embeded\python.exe -s .\tools\quant_mixed.py --help
```

Pares a reconverter e contra o que comparar:

| conversor | fonte | comparar com |
|---|---|---|
| `quant_w4a4` | `hunyuanvideo1.5_720p_t2v_fp16` | `hunyuanvideo1.5_720p_t2v_fp16_w4a4_convrot.safetensors` |
| `quant_w4a8` | `hunyuanvideo1.5_720p_t2v_fp16` | `hv15_w4a8.safetensors` |
| `quant_mixed` | `wan2.1_vace_1.3B_fp16` | `wan21-vace-13b-misto005.safetensors` (usar o `promote_error` do sidecar) |
| `quant_mixed` | `beyond-reality-zimage-v2_native` | `zimage-v2-mixed.safetensors` |
| `quant_int8` | qualquer fonte com saida int8 no disco | a saida correspondente |
| `quant_w4a4_smooth` | -- | **nao ha saida no disco**; ver abaixo |
| `svdq_to_bf16` | -- | **nao ha saida no disco**; ver abaixo |

**Cada reconversao precisa dos MESMOS parametros da original.** Eles estao no `.quant.json` ao
lado de cada saida: `convrot_groupsize`, `group_size`, `selection.promote_error`,
`selection.sigma_weight`. Reconverter com parametro diferente produz arquivo diferente e isso nao
seria um achado, seria um erro de operacao.

Comparar:

```bash
.\python_embeded\python.exe -s -c "import hashlib,sys;h=lambda p:__import__('hashlib').sha256(open(p,'rb').read()).hexdigest();print(h(sys.argv[1])==h(sys.argv[2]))" A B
```

(para arquivos grandes, ler em blocos -- o `verificar_migracao.py` ja tem a funcao)

## Os dois sem saida no disco

`quant_w4a4_smooth` e `svdq_to_bf16` nunca escreveram nada que ainda esteja aqui, entao nao ha
contra o que comparar. Para esses a aceitacao e mais fraca e precisa ser dita como tal:

- rodar uma conversao pequena e **carregar a saida pelo loader normal do ComfyUI**;
- `tools/verify_w4a4.py <saida> --source <fonte> --kernel-smoke` no caso do smooth;
- e registrar que a verificacao foi "carrega e despacha", **nao** "byte a byte".

O `smooth` e o mais importante dos dois, porque foi o unico dos sete que nunca teve guarda de RAM
nem de disco e ganhou as duas nesta migracao -- codigo novo num caminho que nunca rodou com ele.

## A pendencia de codigo: FECHADA no mesmo dia

`svdq_to_bf16.write_checkpoint()` foi **apagada** -- 82 linhas, mais `COPY_CHUNK` e `DTYPE_NAMES`,
que so ela usava. O `test_svdq_write_contract.py` foi redirecionado para
`_conversion.Conversion.commit` antes da remocao, e passa 24/24.

Um dos tres cenarios daquele teste **inverteu de proposito**, e vale saber por que: ele afirmava
que o `finally` do escritor apagava um `.partial` pre-existente. O nucleo abre o `.partial` FORA
do `try` justamente para que o `FileExistsError` nao alcance o `finally` -- o arquivo de um
processo que esteja escrevendo AGORA sobrevive. O teste agora afirma a sobrevivencia.

As quatro assercoes de "o contrato continua inteiro" (fsync antes do replace, rename atomico,
limpeza em todo caminho, escrito == planejado) tambem mudaram de alvo: procuravam os simbolos
dentro de `svdq_to_bf16.py` e agora procuram no nucleo. Deixaram de valer para um arquivo so e
passaram a valer para os sete de uma vez.

## Nao coberto por este plano

Nenhuma imagem. Byte-identidade prova que a migracao nao mudou a saida; **nao** prova que a saida
presta -- isso nunca foi verdade nesta bancada e continua nao sendo. Para qualidade, o caminho e
`tools/avaliar_referencia.py` no braco nao quantizado e depois um render olhado por alguem.

---

# RESULTADO -- executado 2026-09-01 na 3090

Criterio acima escrito antes. Nada dele foi mexido depois de ver numero.

## Passou: quatro pares, sha256 do arquivo inteiro

| conversor | fonte | bytes | sha256 |
|---|---|---|---|
| `quant_w4a4` | hunyuanvideo1.5 fp16 | 8 507 690 240 | `A3485DAA…A4732FBF` |
| `quant_w4a8` | hunyuanvideo1.5 fp16 | 8 847 567 376 | `3ED43444…A41E76D7` |
| `quant_mixed` | wan2.1 vace 1.3B | 2 310 437 144 | `212B9111…0B40A142` |
| `quant_mixed` | beyond-reality-zimage-v2 | 3 403 133 032 | `4463AC4E…2113CF5C` |

O par do Z-Image nao estava planejado e foi acrescentado porque, sem ele, o `quant_mixed` ficaria
provado numa arquitetura so. Wan seleciona 2 w4a4 / 298 w4a8; Z-Image seleciona 117 / 53 -- ramos
bem diferentes do mesmo codigo.

Parametros vieram do `.quant.json` de cada saida, como o plano exigia. O do Wan precisou tambem da
analise (`bench/wan21_mixed.analysis.json`), casada por `source_identity_sha256` e nao por nome.

## Nao deu para reconverter: as tres saidas Z-Image pre-proveniencia

`zimage-v2-mixed`, `zimage-v2-w4a4` e `zimage-v2-mixed-t0.*` foram feitos com analises anteriores a
2026-08-22, sem `source_identity_sha256`. O `quant_mixed` recusa uma analise sem chave de
proveniencia, em vez de pular a checagem -- que era como a guarda velha passava. Reproduzi-los exige
recalibrar. **Isto nao e uma reprova**: e a guarda funcionando, e esta escrito para que ninguem leia
"nao reconverti" como "reconverti e deu diferente".

## Aceitacao mais fraca, dita como tal

| conversor | o que rodou | resultado |
|---|---|---|
| `quant_int8` | conversao real (10,45 GiB) + loader normal + contagem de despacho | 432 modulos `int8_tensorwise`, 8/8 forwards quantizados, 0 dequantize, `comfy_kitchen.backends.cuda` |
| `svdq_to_bf16` | recuperou 11,46 GiB do `svdq-int4_r32-z-image-turbo` + loader normal | carregou como `Lumina2`, 6 154 908 736 params, bf16, 34 qkv fundidas |

Nenhuma das duas compara contra saida anterior, porque nao existe uma. "Carrega e despacha" nao e
"byte a byte".

## `quant_w4a4_smooth`: NAO rodou, e o motivo nao e o codigo

As duas entradas exigidas nao estao nesta maquina -- nao ha `gemma_3_12B_it_heretic.safetensors`
(BF16) nem nenhum Gemma `convrot_w4a4` para `--calibrate-with`. Busca recursiva com `-Force` nos
dois roots, que enxerga `.disabled`.

**O BF16 ja esteve aqui, e isso importa para o custo.** O dono lembrava que "o gemma inteiro nunca
esteve aqui"; o sidecar diz o contrario e e evidencia direta:
`gemma_3_12B_it_heretic_w4a8.quant.json` registra `source_size = 23 545 681 250` (21,93 GiB) lido
de `text_encoders/gemma_3_12B_it_heretic.safetensors`, numa conversao de 25,5 s **nesta 3090**,
cuja saida de 7,53 GiB esta no disco ate hoje. Entao rebaixar nao e apostar num arquivo hipotetico:
e restaurar um estado que ja funcionou. (O `os error 1455` que o `CLAUDE.md` registra ao mapear
"a fonte Gemma de 21,93 GiB" e o mesmo arquivo.)

Duas saidas foram tentadas, as duas fecharam por motivo medido:

- fonte fp8 + calibragem no gemeo w4a8: calibragem rodou inteira (96 normas) e morreu depois com
  `KeyError: 'F8_E4M3'`;
- Gemma-3 1B e fonte valida mas nao serve de calibragem: `load_clip` o detecta como `lumina2` e o
  tokenizer levanta `invalid tokenizer`.

Entao `conv.guard()` e `conv.commit()` do `smooth` **continuam sem ter rodado**. Destravar custa um
download de ~22 GiB mais uma conversao W4A4 dele -- decisao do dono.

## Dois defeitos reais achados pelo caso de CONTROLE

O teste de recusas do `smooth` tem um controle negativo: argumentos validos tem de PASSAR da guarda.
Sem ele, um conversor que morresse em toda invocacao passaria em todas as recusas. **O controle
falhou, e a falha era o achado.**

1. **Sem guarda de "zero camadas".** Apontado para um Wan, o `smooth` imprimia `Layers: 0` e saia
   com rc=0 -- um `--dry-run` respondendo SUCESSO para uma conversao sem nada a converter. Os outros
   quatro ja recusavam.
2. **Sem guarda de dtype.** Validava nomes, nunca o dtype, e morria ~3 min depois dentro do
   `read_tensor` de outro arquivo. Agora recusa em **1,8 s**, e o teste cobra esse tempo.

`tools/test_smooth_guards.py`, novo: 7/7. Suite completa depois das edicoes: **13 suites, 13 exit 0**.

## Coisa que so a corrida ensina

`convrot_groupsize` tem de ser potencia de **4**, nao de 2: 128 levanta `Regular Hadamard size must
be a power of 4` em `comfy_kitchen/tensor/int8_utils.py:22`. Explica 64 e 256 serem os unicos
valores usados aqui. O preflight de backend pegou antes de qualquer trabalho.

## Adendo 2026-09-16 -- as duas saidas da "aceitacao mais fraca" agora tem sha256 registrado

A tabela de aceitacao fraca acima existe porque `quant_int8` e `svdq_to_bf16` **nao tinham saida
anterior no disco** contra a qual comparar. As saidas que eles escreveram em 2026-09-01 ficaram em
`F:/_migracao/`, fora de toda raiz declarada de modelo -- o ComfyUI nunca as enxergou -- e nenhum
documento desta arvore as citava pelo nome nem pelo tamanho.

Medido em 2026-09-16, antes de qualquer decisao sobre elas:

| conversor | saida de 2026-09-01 | bytes | sha256 |
|---|---|---|---|
| `quant_int8` | `hv15_int8_convrot.safetensors` | 11 225 609 256 | `0b8a704aa664f450b1118a25b457c222a4b03bc0ca7feaa6ccc6ad9653a6dad3` |
| `svdq_to_bf16` | `zimage_turbo_recuperado_bf16.safetensors` | 12 309 874 184 | `ba4392077d3d3e25ebc8306ebfbcfb2b8229c741760de0c84142c6d433cf566e` |

**Com estes dois numeros, os sete conversores passam a ter referencia de identidade byte a byte** --
nao so os quatro pares fortes da tabela de cima. Uma proxima migracao do nucleo reconverte e compara
o sha, em vez de aceitar com o "converteu e carregou", que e o que a tabela fraca registrava.

Como reconverter, se for preciso:

- `quant_int8`: fonte `ComfyUI/models/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors`
  (16 653 368 128 B, no disco), `--convrot`, `convrot_groupsize 256`, arquitetura `hunyuan_video_15`.
  Tudo isso esta no `.quant.json` ao lado. A conversao original levou **23,4 s**.
- `svdq_to_bf16`: fonte `ComfyUI/models/diffusion_models/svdq-int4_r32-z-image-turbo.safetensors`
  (3,36 GiB, no disco). **Esta saida nao tem sidecar**; os unicos parametros conhecidos sao os que o
  proprio `__metadata__` do arquivo carregava (`dequantized_from`, `dequantized_note`, 521 tensores,
  todos BF16). Tempo da conversao original: nao registrado.

**Nao coberto.** O sha de uma saida so vale como referencia se a FONTE tambem for a mesma: o sidecar
do int8 grava `source_size`, nao `source_identity_sha256`, entao a fonte esta casada por tamanho e
nome, nao por hash -- mais fraco que os quatro pares fortes, e dito aqui para nao ser lido como
igual a eles. A do `svdq_to_bf16` nao tem sidecar nenhum. E identidade byte a byte nunca foi
afirmacao sobre qualidade: nenhuma das duas saidas foi medida na saida, em nenhum render.
