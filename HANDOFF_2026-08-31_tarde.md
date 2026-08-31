# Handoff — 2026-08-31, tarde

Continuação de [HANDOFF_2026-08-31.md](HANDOFF_2026-08-31.md), que cobre a manhã. **Tudo aqui foi
executado nesta bancada hoje**, salvo onde diz TRAÇADO. Onze commits, `f0e2c1f` a `fa68a39`.

Resumo dos modelos e das ferramentas em [RESUMO_MODELOS_2026-08-31.md](RESUMO_MODELOS_2026-08-31.md).

## Estado ao fechar

```
GPU 0  RTX 3090      livre
GPU 1  RTX 3080 Ti   livre (usada hoje, cc 8.6 igual)
F:\GPU_BENCH.lock    livre
disco F:             ~300 GB livres
git                  limpo
tickets .scratch     29 de 29 fechados
```

## Os três achados

### 1. Text encoder quantizado no ComfyUI não faz conta quantizada. Nenhum.

Contado numa codificação real, não lido:

```
100/100 Linear   MixedPrecisionOps.Linear, quant_format convrot_w4a4, peso QuantizedTensor
_convrot_w4a4_forward        0
QuantizedTensor.dequantize 350
```

O peso fica 4 bits na VRAM e a matemática é BF16 dequantizada. **Economia de memória, não de tempo.**

A trava que morde **não** é a óbvia. `comfy/sd1_clip.py:114` fixa `full_precision_mm=True` para todo
text encoder; virar só ela não muda nada. Instrumentando cada termo de `_use_quantized`
(`comfy/ops.py:1372-1377`) aparece `comfy_force_cast_weights=True`, que vem de **`comfy/sd.py:269`,
`set_model_compute_dtype(torch.float32)`, aplicado a TODO objeto CLIP**. São duas travas
independentes.

Isso vale para o `gemma` deste projeto — 336 camadas, as duas travas `True`, 336 dequantize, zero
forward quantizado. **O carro-chefe do conversor é economia de memória e nada mais.** O modelo de
difusão é outro caminho e não sofre: Z-Image mediu 340 quantizados e 0 dequantize.

### 2. Destravar paga ou não conforme o COMPRIMENTO DO PROMPT

Mesmo arquivo, mesma placa, só o prompt muda:

```
qwen_3_4b W4A4, 3080 Ti, mediana de 3
tokens   travado  destravado
    22     80,7      120,1    1,49x MAIS LENTO
    75    100,1      105,8    1,06x MAIS LENTO
   199    151,9       95,2    1,60x mais rápido   <- cruza entre 75 e 199
   424    245,2      102,0    2,40x
   850    456,1      124,3    3,67x
  1496    824,5      249,0    3,31x
```

O caminho destravado quase não se move de 22 a 424 tokens; o travado sobe com a sequência. Prompt
real fica acima do cruzamento — em 850 tokens o encoder quantizado bate até o **BF16 original**.

Nos três encoders medidos, com o custo em precisão:

```
encoder                formato  quant     travado  destravado   tempo              erro C-vs-B   cos
qwen_3_4b (4B)           W4A4  2,4 GiB    70,2 ms    82,9 ms   1,18x MAIS LENTO   5,99e-1   0,949
gemma_3_12B_heretic      W4A8  8,1 GiB  1772,3 ms   478,9 ms   3,70x mais rápido  2,11e-1   0,982
qwen3vl_32b_minimax      W4A4 13,2 GiB   311,6 ms   117,7 ms   2,65x mais rápido  9,73e-2   0,99989
```

**Ler a coluna de erro com cuidado:** é erro relativo do *condicionamento* contra o próprio braço
travado, não segundos e não qualidade de imagem. Só o Qwen tem gêmeo BF16 no disco, e nele dá para
separar: o peso de 4 bits sozinho custa 1,44e-1, destravar leva a 6,09e-1.

### 3. O que os checkpoints públicos realmente executam

Com os pesos na GPU, registrando a implementação que o registry escolheu:

```
checkpoint                                 linear_dtype  impl     flag muda?  ramo
zimage-v2-w4a4                 (nosso)         int4      cuda        SIM      nativo int4
LTX25-distilled-DiT-comfy-w4a4 (riftcast)      int4      cuda        SIM      nativo int4
MiniMax_H3_FL2VA         (Abiray, 791k dl)     int8      cuda        NÃO      INT8, por instrução
```

O da Abiray **executa matemática quantizada** — só nunca toma o ramo int4. O resumo do próprio
arquivo se contradiz: `"linear_dtype": "int4"` no topo e `"w4a4_int4mm_layers": 0` três chaves
abaixo, com as 117 camadas dizendo `"int8"`. **Ler as camadas, não o resumo.**

Segundo W4A4 público que roda 4 bits de verdade: `LTX25-distilled-DiT-comfy-w4a4`,
`quantized_by: riftcast/ltx25-quant-lab`. **Sétimo escritor conhecido** para o ticket 08.

### E dois arquivos públicos não abrem sem dynamic-VRAM

Os dois da Abiray carregam bytes **depois do último tensor** (83 e 64), e `safetensors.safe_open`
recusa com `incomplete metadata, file not fully covered`. Não é download quebrado: o
`Content-Length` do servidor bate byte a byte e um `Range: bytes=-83` devolve a mesma cauda. O
leitor do dynamic-VRAM (`comfy_aimdo.model_mmap`) aceita. Quem passa `--disable-dynamic-vram` —
necessário aqui para Nunchaku e LTX 2.5 — não carrega, com um erro que culpa o arquivo.

## O critério por sigma: implementado, medido, indistinguível

`calibrate_activations.py` grava agora `sample_sigma` (um sigma por linha amostrada, dos mesmos
slots do reservoir) e `quant_mixed.py` aceita `--sigma-weight none|sigma|sigma2|high`, **default
`none`** para não reinterpretar análise já gravada. 15 testes sem GPU em
`tools/test_quant_mixed_sigma.py`.

Oito sementes, desenho pareado, **mesmo orçamento de 56 promoções nos três braços**:

```
epsilon médio, 8 sementes     média       min       max    espalhamento entre sementes
plano56                     1,3326e-1  1,1220e-1 1,5577e-1        1,388x
sigma2                      1,3684e-1  1,1731e-1 1,5216e-1        1,297x
sigma_alto                  1,3685e-1  1,2276e-1 1,4695e-1        1,197x

diferença pareada contra o plano:  sigma2     +3,37%  (erro-padrão 3,95%,  vence 4/8)
                                   sigma_alto +3,46%  (erro-padrão 3,93%,  vence 2/8)
```

Espalhamento dentro de um braço: **1,39x**. Diferença entre critérios: **3,4%**. O efeito está
dentro do próprio ruído. Vencedor por semente 3/3/2.

Isso **corrige** a primeira versão, que reportou "plano vence 3/3" em três sementes e dois braços —
artefato de amostra pequena.

## Meus erros de hoje, e o que pegou cada um

| erro | o que pegou |
|---|---|
| Hipótese de "PR de uma linha" no ramo QWEN3VL_32B | a execução: carrega e codifica normalmente |
| Veredito impresso sobre um forward que **não rodou** (MiniMax quer lista de latentes) | ninguém, na hora — corrigido no probe depois |
| `--forward-only` chamando módulos **na CPU**, medindo o backend eager | `ValueError: Expected a cuda device` ao instrumentar |
| Destravamento por escrita no módulo, apagado por `patch_model` a cada load | o contador de dispatch: 350 numa run, 0 na seguinte |
| Confundidor de orçamento no A/B do sigma (56 contra 53 camadas) | reconstruir o braço plano no mesmo orçamento |
| Mecanismo do cruzamento afirmado sem medir ("escala com o peso") | os próprios números: 70,2 / 1772,3 / 311,6 ms não é monotônico |
| Prompt curto na 3090 e longo na 3080 Ti — dois eixos | refeito na mesma placa antes de concluir |
| Tomei o lock antes de um benchmark que toma sozinho | a própria ferramenta recusou a run |
| Serializei um condicionamento de 192 milhões de floats com `tolist()` | 32 GiB num processo, morto antes de atrapalhar |

## Aberto

| item | estado | de quem |
|---|---|---|
| **Render LTX real com o Gemma destravado** | 3,70x no encode está medido; s/it e qualidade não | aqui |
| **Tamanho do peso como eixo** do cruzamento | o prompt foi isolado, esse não | aqui |
| **LoRA sobre peso de 4 bits custa qualidade?** | WARN antigo do preflight, nunca medido | aqui |
| **`--sigma-weight sigma`** (linear) | testado como aritmética, nunca virou checkpoint | aqui |
| **Fidelidade do encoder Winnougan** | sem gêmeo BF16 nesta bancada | bloqueado |
| **Nobreak** | a queda de 01:31 já custou uma noite | dono |
| **AnyDesk desassistido** | ID **227311266** | dono |
| **Ticket 08** | sete escritores conhecidos agora | dono |
| **Proxies `.cn`** | melhoraria o garimpo do posto de escuta | dono |

## Não coberto, em geral

Nenhuma métrica perceptual em lugar nenhum. Uma placa por medição (sm86 as duas). Sem SASS — "ramo
nativo" sempre significa "produz número diferente do fallback", nunca "a instrução foi observada
emitindo". As travas do text encoder são soltas por monkeypatch pós-load, que **não é um caminho que
o ComfyUI ofereça a um usuário**: `custom_operations` em `model_options` é a única saída real e
nenhum nó a expõe.
