# `svdq_to_bf16.py`: três achados de baixa confiança seguem sem fechar

Type: task
Status: open

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
