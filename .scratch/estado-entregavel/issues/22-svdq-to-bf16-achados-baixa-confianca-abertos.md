# `svdq_to_bf16.py`: três achados de baixa confiança seguem sem fechar

Type: task
Status: resolved

## Question

`AUDITORIA_2026-08-18.md` seção 5 lista achados que o próprio auditor marcou como não
demonstrados o bastante para entrar na fila de conserto ("não entram na fila de conserto até
alguém fechar a lacuna nomeada"). O gate de NaN (achado grave, item 12) **já foi corrigido**
(`zero_leak > 1e-3` virou `not (zero_leak <= 1e-3)` + `torch.isfinite`, reconferido nesta triagem).
Estes três, não:

1. **`split_fused` guards de shape não verificam a fusão em si** (linha 267 segundo
   `AUDITORIA:190`). Os dois cenários que quebrariam (GQA com q/k/v de tamanhos diferentes somando
   um total divisível por 3; `FeedForward` diffusers com `net.0.proj` como projeção única) são
   construídos, não observados num arquivo real. Nos dois Z-Image desta máquina as shapes conferem
   (`to_qkv` out=11520=3·3840, `net.0.proj` out=20480=2·10240) — **não morde hoje**, mas o guard
   não impede o cenário construído.

2. **Bias fundido mantém o nome antigo no passthrough** (linha 265 segundo `AUDITORIA:191`). Os
   136 stems quantizados dos dois arquivos Z-Image aqui têm zero bias — não observado ativo. O
   Qwen-Image tem bias (`transformer_blocks.0.attn.to_qkv.bias`), mas escapa porque o padrão
   testado é `attention.to_qkv` e o Qwen usa `attn.to_qkv` — soma-se ao "não morde hoje" só por
   coincidência de nomenclatura, não por proteção.

3. **Dtype BF16 hardcoded para as norms** (linha 252 segundo `AUDITORIA:192`). Confirmado que num
   Gemma 12B real as norms são BF16, então hoje coincide. Um Gemma-3 exportado direto do HF com
   norms em F32 (cenário comum, não observado nesta máquina) sairia errado sem aviso.

Também aberto, mais teórico: a afirmação do docstring de que a sonda é "exata, sem erro nenhum"
segue overclaiming — o erro de arredondamento BF16 na escala (~1,4e-3 relativo médio) já está
registrado no `CLAUDE.md` como fato corrigido na prosa, mas a confirmação definitiva (quantizar um
peso conhecido com o quantizador do nunchaku de verdade e comparar) pede GPU e não foi feita.

## Critério de fechamento

Não fecha com decisão escrita — mas nem todo item aqui pede código. Fecha quando, para cada um
dos 3 itens:
- for adicionado um guard/assert que recusa explicitamente o cenário perigoso em vez de assumir
  silenciosamente que ele não vai acontecer (para os itens 1 e 3); ou
- for corrigido o nome do bias fundido no passthrough para refletir a fusão real (item 2);

**ou**, alternativamente, para qualquer item onde a correção certa dependa de ver o comportamento
real do nunchaku (o item 1 do gemma-3-F32, ou a confirmação numérica da sonda) — fica marcado
`[GPU]` explicitamente no arquivo, igual ao padrão que o resto do projeto já usa, em vez de ficar
como uma alegação categórica não confirmada no docstring.

## Resolucao

Os tres itens fecharam pelo primeiro ramo do criterio -- guard que **recusa** o cenario perigoso
em vez de assumir que ele nao acontece -- e o quarto (o overclaim do docstring) ja estava
corrigido antes desta rodada e so ganhou a marca `[GPU]`.

### O que mudou no codigo

`tools/svdq_to_bf16.py` ganhou `check_fusion(header, stems, config)` e o `_in_out()` que ela usa,
chamada em `main()` **antes de qualquer tensor ser lido** (e pulada sob `--keep-fused`, porque
nesse modo nada e cortado). Quando ela acha algo, o tool imprime cada problema e devolve 1 sem
escrever nada.

Quatro recusas, uma por cenario:

1. **stem fundido que carrega bias** (item 2 do ticket) -- o peso vira `to_q/to_k/to_v.weight` e
   o `to_qkv.bias` atravessa o passthrough com o nome fundido, entao nenhum peso casa com ele.
2. **`to_qkv` com `out != 3*in`** (item 1, cenario GQA).
3. **`n_kv_heads != n_heads` no config do proprio modelo** (item 1, dito pelo publicador).
4. **`net.0.proj` que e projecao unica e nao par de gates** (item 1, cenario `FeedForward`
   diffusers) -- decidido pelo irmao `net.2`: par de gates alimenta o `w2` com metade da saida;
   projecao unica alimenta com a saida inteira.

### O que a medicao mudou em relacao ao ticket

**O item 3 estava errado como escrito.** A auditoria dizia "dtype BF16 hardcoded para as norms" e
que um Gemma-3 com norms F32 "sairia errado sem aviso". Nao sai: as norms sao **passthrough**,
copiadas byte a byte com o `info["dtype"]` do proprio header (`svdq_to_bf16.py`, laco de
`entries`/`("copy", info)`), e o caminho de copia nunca converte. Medido em quatro arquivos
(z-image-turbo, beyond-reality-zimage-v2, qwen-image, flux.1-dev): 283/283, 283/283, 973/973 e
1844/1844 tensores de passthrough sao BF16, e o dtype seria preservado de qualquer jeito. O que e
de fato hardcoded e o dtype com que a camada Nunchaku e construida para a **recuperacao** --
outra coisa, e agora dito no comentario ao lado da linha. Regra do `CLAUDE.md`: quando a medicao
contradiz a alegacao, a alegacao e o que muda.

**Os itens 1 e 2 sao mais graves do que "latente".** A auditoria os chamou de cenarios
construidos, nao observados. Estao observados, no `svdq-int4_r128-qwen-image.safetensors` desta
maquina:

- os **180** stems do Qwen que estao a um prefixo de casar o padrao (`attn.to_qkv` em vez de
  `attention.to_qkv`; `img_mlp.net.0.proj`/`txt_mlp.net.0.proj` em vez de
  `feed_forward.net.0.proj`) **todos** carregam bias -- contra **zero** bias nos 102 stems que
  casam em cada Z-Image;
- e o `net.0.proj` do Qwen **nao e** par de gates: `out=12288` e o `net.2` irmao consome os
  12288 inteiros, nao 6144. E exatamente o cenario "projecao unica" que a auditoria descreveu como
  hipotetico.

Ou seja: os dois viram ativos **no dia em que alguem alargar o padrao para suportar Qwen**, que e
o proximo passo obvio. Nao e defeito adormecido; e defeito armado.

### Comandos exatos (EXECUTADOS, 2026-08-21)

```
.\python_embeded\python.exe -s <scratch>\svdq_fusion_evidence.py
.\python_embeded\python.exe -s <scratch>\check_fusion_against_real_files.py   -> EXIT=0
.\python_embeded\python.exe -s -m py_compile tools\svdq_to_bf16.py
.\python_embeded\python.exe -s <scratch>\run_cpu_tests_only.py tools\test_svdq_verify.py
    -> 12 test_* function(s): 12 passed, 0 failed.
```

`check_fusion_against_real_files.py` roda duas passadas por arquivo. **Como esta:** zero problema
nos quatro -- nenhum falso positivo, nenhuma conversao que hoje funciona seria recusada. **Com o
padrao alargado** (nomes do Qwen renomeados para o que `split_fused` casa, shapes reais
preservadas): 180 recusas de bias fundido e 60 de projecao unica. E o unico jeito de exercitar os
guards com dados reais, ja que nenhum arquivo que casa o padrao hoje tem o defeito.

### O que ficou sem cobertura

- **So header.** Nenhuma conversao foi feita, nenhum peso recuperado, o kernel do nunchaku nao
  rodou nesta verificacao. Os guards decidem por shape e nome declarados; nao distinguem um corte
  correto de um plausivel, so recusam os provadamente errados.
- **O cenario GQA continua nao observado.** Nenhum dos quatro arquivos tem `n_kv_heads <
  n_heads`; os guards 2 e 3 estao escritos a partir do mecanismo, nao de um arquivo que morde.
  Marcado como tal no docstring da `check_fusion`.
- **Numero a nao ler errado:** na passada alargada o Qwen aparece com 480 stems e nao 600, porque
  meu rename colapsa `img_mlp` e `txt_mlp` no mesmo nome. Um alargamento de verdade os manteria
  distintos, e as recusas seriam 120 e nao 60 na parte de MLP. O guard nao muda; a contagem do
  ensaio, sim.
- **A confirmacao numerica da sonda continua `[GPU]`** e nao foi feita: quantizar um peso BF16
  conhecido com o quantizador do nunchaku e comparar contra o dequant em fp32. O docstring ja
  dizia "Closing it needs a GPU"; agora carrega a marca `[GPU]` explicita que o resto do projeto
  usa.
