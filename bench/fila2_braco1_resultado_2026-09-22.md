# O braço 1 existe e fecha 60,06% da distância PTQ → Bonsai. E o controle refutou a própria previsão

Critério em `bench/criterio_fila_gpu_compensacao_2026-09-22.md`, escrito antes de qualquer número e
**não alterado**. Lock tomado como `bench:klein4b_braco1_diffusers`, 3090 em 2 MiB / 0% no início.
Oito sementes, 4 braços, trajetória do BF16 imposta a todos, 8 passos, 512 px, `cfg 1.0`,
euler/simple, encoder `qwen_3_4b` com `--clip-type flux2`.

O braço 1 estava bloqueado na fila 1 pelas cinco somas in-place do flux do ComfyUI. **O bloqueio
dissolveu quando ele perguntou por que eu não tinha lido o RUNTIME do Bonsai**: o
`prism-image-studio` deles carrega o modelo com `Flux2Transformer2DModel.from_config`, ou seja
**diffusers**, e o `Flux2Transformer2DModel` do diffusers 0.38.0 tem zero somas in-place. Não era
preciso patch no core nem monkeypatch. Cinco tentativas minhas brigaram com a implementação errada.

---

## 1. O placar do critério, previsão por previsão

    P1  eps(braco0) >= 3x eps(braco2)            CONFIRMADA   3,4946x     (refutava < 1,5x)
    P2  braco1 fecha >= 25% da distancia         CONFIRMADA  60,06%       (min 58,31 max 62,06)
    P3  8 sementes, pareado, ep, placar          HONRADA      8 sementes  (a fila 1 entregou 1)
    P4  o ganho se concentra em sigma ALTO       REFUTADA    56,84% alto contra 63,23% baixo
    P5  bracos 0 e 1 diferem SO no denso         CONFIRMADA  140/149 byte a byte identicos

    controle `zero`    peso identico ao braco 0  PASSOU      169/169 byte a byte
    controle `ruido`   tem de PIORAR             REFUTADA    melhorou 0,39%, e ver secao 3

## 2. O número, com o espalhamento ao lado porque P3 exige

Média das 8 sementes do rel-RMSE do epsilon contra o BF16, entradas casadas:

    braco                  media         min         max   espalha   fracao braco0->braco2
    braco0_ptq        9.8810e-01  9.7938e-01  1.0000e+00    1,021x      0%   (o que perde)
    braco1_ruido      9.8425e-01  9.7601e-01  9.9595e-01    1,020x      0,55%   <- controle
    braco1_ajuste     5.6463e-01  5.4611e-01  5.7445e-01    1,052x     60,06%  <- a HIPOTESE
    braco2_bonsai     2.8289e-01  2.7351e-01  2.9434e-01    1,076x    100%   (o que ganha)

Diferença **pareada** contra o braço 0 — pareada porque dentro de uma semente todos os braços
recebem a mesma trajetória imposta, então a diferença não carrega a variação da semente:

    braco1_ajuste   d +4,2348e-01   erro-padrao 2,2463e-03   188,5x o ep   razao 1,7503x   vence 8/8
    braco1_ruido    d +3,8558e-03   erro-padrao 1,0858e-04    35,5x o ep   razao 1,0039x   vence 8/8
    braco2_bonsai   d +7,0521e-01   erro-padrao 2,3960e-03   294,3x o ep   razao 3,4946x   vence 8/8

**O espalhamento entre sementes dentro de um braço é 1,02x a 1,08x**, contra o **1,39x** que este
repo mediu no Z-Image. É por isso que o efeito de 1,75x está a 188x o próprio erro-padrão: a
condição que mataria a P2 — a diferença 0-vs-1 caber no espalhamento do braço 0 — está a duas ordens
de grandeza de acontecer.

Passos vencidos: **braço 2 ganha 64 de 64**. O teto não foi alcançado por ninguém.

## 3. O controle `ruido` refutou a própria previsão, e no sinal oposto

O critério dizia: *"perturbar o conjunto denso com ruído do mesmo tamanho do ajuste, em vez de
ajustar. Tem de **piorar**."* Não piorou. **Melhorou 0,39%**, e não é ruído de medição: 35,5x o
erro-padrão pareado, vencendo 8/8. Eu chamei isso de "dentro do ruído" numa leitura anterior das
três primeiras sementes, e estava errado — é pequeno e sistemático.

O ruído foi casado com o ajuste **tensor a tensor**, pelo desvio rel-L2 que o ajuste MEDIU, não por
`lr × épocas`, que é uma proxy que o Adam não respeita (passo normalizado, momento, 30 épocas de
direção coerente). Maior desvio: **9,015e-02** contra **9,010e-02** do ajuste — casado a 0,05%. A
ferramenta agora **recusa** o modo ruído sem `--casar-com <json do ajuste>`.

**O que o controle mostra, então, é mais forte do que o que ele previa.** Com a magnitude da
perturbação casada nos nove tensores, a direção vale **109x** mais que a magnitude:

    fracao da distancia braco0 -> braco2
      braco1_ajuste  60,06%
      braco1_ruido    0,55%      -> 109x

Se o ganho fosse "perturbar o denso funciona", o ruído teria fechado uma fração comparável. Fechou
meio por cento.

**E o ruído tem uma assinatura invertida em sigma**, que só apareceu por a P4 pedir a separação:

    faixa            braco1_ajuste   braco1_ruido
    sigma ALTO            56,84%         0,95%     (48,7x o ep)
    sigma BAIXO           63,23%         0,15%     (3,9x o ep, vence 7/8, min -0,00%)

O pouco que o ruído compra está em **sigma alto**; o ajuste ganha mais em **sigma baixo**. Não tenho
mecanismo medido para isso e não invento um: registro como observação de 8 sementes num prompt.

## 4. P4 refutada pela segunda vez, na mesma direção

Eu previ o ganho do braço 1 concentrado em sigma alto, porque é lá que a estrutura é decidida e onde
o denso (modulação, embedders) mandaria mais. Medido: **56,84% em sigma alto contra 63,23% em sigma
baixo**, e a razão do ajuste sobre o braço 0 vai de 1,5905x (alto) a 1,9870x (baixo).

Isso é a **segunda** refutação da mesma previsão nesta investigação — a fila 1 já tinha medido o
ganho do Bonsai maior em sigma baixo (4,6710x contra 2,9669x). Dois braços independentes, o mesmo
sinal contra a minha intuição. O que essa intuição tinha de errado, [JULGAMENTO]: em sigma alto o
erro do PTQ é **saturado** (rel-RMSE ~1,07, ou seja a previsão está tão longe do BF16 quanto o
próprio sinal), e não há o que compensar num regime saturado. O que me faria mudar de ideia: medir
um PTQ menos agressivo, onde sigma alto não sature, e ver se o ganho migra para lá.

## 5. Onde o gradiente escolheu mexer

Desvio rel-L2 por tensor depois de 30 épocas, do maior ao menor:

    9,0099e-02  time_guidance_embed.timestep_embedder.linear_2.weight
    5,2625e-02  time_guidance_embed.timestep_embedder.linear_1.weight
    4,7131e-02  double_stream_modulation_img.linear.weight
    3,3231e-02  double_stream_modulation_txt.linear.weight
    2,6600e-02  norm_out.linear.weight
    2,2217e-02  proj_out.weight
    1,6924e-02  single_stream_modulation.linear.weight
    1,2497e-02  context_embedder.weight
    5,6844e-03  x_embedder.weight

O **embedder de timestep** foi escolhido como o que mais precisa mudar, não a modulação — e ele
alimenta *toda* a modulação. Isso é consistente com o mecanismo que o mapa do Bonsai otimiza ao
deixar a modulação em alta precisão, mas é **um ajuste, um modelo**, não uma afirmação geral.

Perda: **1,1361e+00 → 3,6883e-01** (3,08x), 64 exemplos (4 prompts × 2 sementes × 8 passos), 30
épocas, Adam lr 1e-5, 9 tensores / 195.035.136 valores = **5,03% do modelo**.

## 6. Correção ao critério: o conjunto denso são 9 tensores, nas duas nomenclaturas

O critério diz 195.042.816 params e a fila 1 corrigiu para "9 tensores em BFL, 69 em diffusers".
**As duas estavam erradas na contagem de tensores em diffusers.** Medido: as 60 `norm_q`/`norm_k`
são `single_transformer_blocks.N.attn.norm_k.weight` — **dentro** de bloco em diffusers também.
Então são **9 tensores, 195.035.136 valores, nas duas nomenclaturas**, e a diferença de 7.680
params contra o critério são exatamente as 60 normas de 128.

## 7. Seis defeitos meus nesta corrida

1. **Classe de pipeline errada.** Escrevi `Flux2Pipeline`; o klein declara `Flux2KleinPipeline` no
   próprio `model_index.json`. As duas existem no diffusers 0.38.0 e usam encoders **diferentes** —
   `_get_mistral_3_small_prompt_embeds` contra `_get_qwen3_prompt_embeds`. O erro resultante não
   menciona pipeline nenhum: morre em `apply_chat_template` dizendo que falta `chat_template`, o que
   manda o leitor consertar o tokenizer. A classe agora sai do `_class_name`.
2. **Faltavam três arquivos do tokenizer** que estão no repo e eu não baixei:
   `chat_template.jinja`, `added_tokens.json`, `special_tokens_map.json`.
3. **`vae=None` mataria a corrida no fim**, com todo o trabalho feito:
   `pipeline_flux2_klein.py:907` lê `self.vae.bn.running_mean` **incondicionalmente** depois do
   loop, inclusive sob `output_type="latent"`. Medido e não assumido: com `vae=None` o
   `vae_scale_factor` cai no fallback 8, que **por sorte** é o valor certo do klein.
4. **O docstring afirmava 69 densos em diffusers** — refutado pela execução da própria ferramenta
   (seção 6).
5. **Um comentário meu citava 3,62 GiB de RAM de captura para 64 exemplos.** Isso foi **deduzido**
   de um shape errado. Medido: **0,50 GiB**. A ferramenta agora imprime o custo em vez de eu
   estimar.
6. **Quebrei o arquivo escrevendo código por script gerador** — `\n` dentro de literal virou quebra
   de linha real, que é a memória `backslash-vira-byte-de-controle` cobrando de novo. Refeito por
   edição direta.

E um que o controle achou sem eu procurar: **o braço 1 perdia a proveniência da origem**, porque
`save_file` sem `metadata=` grava `None`. Foi o que fez o controle `zero` sair com 169/169 tensores
idênticos e o **arquivo** diferindo do braço 0 por 32 bytes — exatamente o `{"format": "pt"}` do
escritor de lá. Consertado.

`tools/compara_checkpoints_byte.py` existe por isso: responde em **três níveis separados** (arquivo,
tensor, estrutura), porque os dois lados foram escritos por escritores diferentes e um veredito de
nível único teria dito "FALHOU byte a byte" e me mandado caçar um defeito inexistente.

## 8. O que isto significa, e o que NÃO significa

**Significa:** um estágio de compensação curto — corpo ternário congelado, só o conjunto denso
ajustado contra a saída do modelo denso — recupera **60% da distância** entre um PTQ ingênuo a 1,58
bit e o Bonsai treinado, mexendo em 5% dos parâmetros com 64 exemplos e 30 épocas numa 3090. Era a
hipótese que o critério chamava de "o achado prático mais útil de toda esta investigação se for
verdade".

**Não significa que é o que o Bonsai fez.** O critério já dizia isso e continua valendo: o braço 2
ganha 64 de 64 passos e continua **1,996x** mais fiel que o braço 1. Sobram 40% da distância que
este ajuste não fecha, e nada aqui diz que mais épocas ou mais dados fechariam.

## Não coberto

- **Nenhuma imagem, em nenhum braço.** Isto mede previsão de epsilon em entradas casadas. Este repo
  já mediu por que a imagem livre não serve para comparar dois quants do mesmo modelo.
- **Um prompt por semente**, 8 passos, 512 px, uma placa, sem métrica perceptual.
- O erro-padrão é da **diferença pareada** e **não é um teste de hipótese** — é a distância do efeito
  ao próprio espalhamento.
- **O braço 1 foi ajustado pelo diffusers e medido pelo ComfyUI.** Esse cruzamento corre *contra* o
  resultado, não a favor, mas nenhum render confirmou que o mesmo peso se comporta igual nos dois
  caminhos.
- **Nenhum número de tempo vale.** O corpo ternário roda desempacotado em bf16, sem kernel de 1,58
  bit; gemlite não está instalado.
- BF16 é o alvo, não a verdade — nunca foi validado contra float32.
- As métricas publicadas do Bonsai (GenEval, HPSv3, DPG) continuam **não replicadas**.
