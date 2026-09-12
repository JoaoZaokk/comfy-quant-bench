# O cadeado do text encoder saiu, e o que ele custava

**Executado 2026-09-12 na 3090.** O dono pediu isso mais de uma vez; eu vinha **repetindo a
medição do obstáculo como se fosse uma limitação aceita** em vez de removê-lo. A frase "o ComfyUI
não deixa" é o começo de uma tarefa, não a conclusão dela.

## O que travava

Duas coisas independentes, e **qualquer uma sozinha basta** para o kernel de 4/8 bits nunca ser
alcançado:

    comfy/sd.py       set_model_compute_dtype(torch.float32)   -> comfy_force_cast_weights = True
    comfy/sd1_clip.py mixed_precision_ops(..., full_precision_mm=True)

O resultado era um text encoder quantizado que **guarda peso de 4 ou 8 bits na VRAM e faz a conta
em BF16 dequantizado**: economiza memória, não economiza tempo, e o kernel nunca roda. Medido
contando chamadas no `gemma_3_12B_it_heretic_w4a8`: **336 de 336 `dequantize`, zero forwards
quantizados.**

## A mudança

Quatro arquivos, 31 linhas. `patches/comfyui_text_encoder_quantized_math.patch` guarda o diff
**porque o repo raiz não rastreia `ComfyUI/`** e `update/update_comfyui.bat` o apagaria em
silêncio.

- `cli_args.py` ganha `--disable-quantized-text-encoder`, que repõe o comportamento antigo.
  **Flag e não variável de ambiente porque existem ZERO `os.environ` em todo o `comfy/*.py`** --
  uma env var seria estranha ao código, e `AGENTS.md` recusa código que não pareça escrito à mão
  para o repositório.
- `ops.quantized_text_encoder_math(quant_config)` decide: solta só quando o checkpoint **de fato**
  carrega camadas quantizadas, e a flag desliga.
- `sd1_clip.py` pergunta a ela em vez de fixar `full_precision_mm=True`.
- `sd.py` pula o upcast para float32 quando o encoder tem matemática quantizada.

**O `sd.py` não pode ler a metadata, e essa foi a minha primeira versão errada.** Toda família de
encoder faz `model_options.copy()` antes de escrever `quantization_metadata`, então a chave
**nunca chega** a quem cria o patcher -- o check leria `None` sempre. A segunda versão perguntou à
árvore de módulos por `_quant_config`, e também falhou: `_quant_config` mora na classe externa
`MixedPrecisionOps`, e a `Linear` aninhada herda de `torch.nn.Module`, não dela. O marcador que a
**instância** carrega é `_full_precision_mm`, posto no `__init__`. Duas tentativas, as duas
pegas pelo contador de forwards -- uma sonda que só comparasse saídas teria dito "soltar não muda
nada" e seria acreditada.

## O A/B, um eixo variado

Cada braço num processo-filho limpo, porque `model_patcher.py:1016` reescreve
`comfy_force_cast_weights` em cada módulo toda vez que o modelo sobe para a GPU.

    braço                    force_cast   fp_mm   forwards quantizados   dequantize
    travado (--disable...)        True     True                     0          336
    destravado (padrão)          False    False                   336            0
                                                impl: w4a8_int8_linear = comfy_kitchen.backends.cuda

## O custo e o ganho, contra o BF16 original

O BF16 deste encoder existe no disco, então a pergunta que decide pôde ser feita: **qual dos dois
braços está mais perto da verdade?** Não "quanto eles diferem entre si" -- os dois usam o mesmo
peso quantizado.

    braço                  rel-RMSE vs BF16   cosseno         ms      GiB
    BF16 original                         -         -   10.404,5    23,55
    travado (antigo)             1,0893e-01  0,994392    1.810,9     7,53
    destravado (novo)            1,2047e-01  0,992906      480,8     7,53

**Soltar custa 1,11x de fidelidade e rende 3,77x de velocidade.** E o quadro inteiro: contra o
BF16, o encoder quantizado com o cadeado solto é **21,6x mais rápido e 3,13x menor**.

Prompt de **1024 tokens**. Isso importa: medido em 2026-08-31 num encoder de 4 B, a virada fica
entre **75 e 199 tokens** -- abaixo dela o caminho destravado é MAIS LENTO, porque o kernel de 4
bits tem custo fixo por camada que um prompt curto não amortiza. Este número vale para prompt
longo, que é o caso real.

## O controle que tinha de passar

Um encoder **não quantizado** não pode mudar. Carregado o `gemma_3_12B_it_heretic` BF16 nos dois
braços: **0 camadas quantizadas, `force_cast_weights` True nos dois**. O upcast para float32
continua onde estava.

E a suíte: `pytest ComfyUI/tests-unit` dá **45 falhas, 1186 passam, 15 puladas, 58 erros** --
**idêntico com e sem a mudança**, medido guardando o diff e rodando de novo. As falhas são de
servidor e segurança, pré-existentes.

## O que isto NÃO cobre

Um prompt, um encoder, uma placa. **Nenhuma imagem**: isto mede o condicionamento, e o que 12% de
erro de condicionamento faz com a foto final é outra pergunta -- e nesta bancada a foto em
trajetória livre já se mostrou incapaz de separar duas quantizações do mesmo modelo. Nada aqui
testa encoder composto (dois ou mais modelos num objeto CLIP), nem o caminho `custom_operations`.
Sem SASS.

## Como refazer

    python_embeded\python.exe -s tools\probe_te_cadeado.py <encoder quantizado> \
        --clip-type LUMINA2 --referencia <o BF16 dele> --controle <um encoder nao quantizado>
