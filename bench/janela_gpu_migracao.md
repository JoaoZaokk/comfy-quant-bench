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

## Uma pendencia de codigo, nomeada

`svdq_to_bf16.write_checkpoint()` saiu do caminho de producao mas continua no arquivo, porque
`test_svdq_write_contract.py` a exercita diretamente em tres cenarios. Redirecionar aquele teste
para `_conversion.Conversion.commit` e entao apagar a funcao. Os tres cenarios ja tem cobertura
equivalente em `test_conversion_core.py`; o redirecionamento e para preservar as anotacoes de
proveniencia daquele arquivo, nao para recuperar cobertura.

## Nao coberto por este plano

Nenhuma imagem. Byte-identidade prova que a migracao nao mudou a saida; **nao** prova que a saida
presta -- isso nunca foi verdade nesta bancada e continua nao sendo. Para qualidade, o caminho e
`tools/avaliar_referencia.py` no braco nao quantizado e depois um render olhado por alguem.
