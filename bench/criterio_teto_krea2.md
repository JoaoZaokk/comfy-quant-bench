# Critério: onde o Krea2 Turbo quebra em W4A4

**Escrito 2026-09-12, ANTES de converter qualquer coisa neste eixo.** As previsões abaixo estão
aqui para poderem ser refutadas; o resultado é anexado no fim sem editá-las.

## A pergunta

A tabela de tolerância do `CLAUDE.md` tem a linha do Krea2 com a coluna `NÃO tolerado` **vazia**:

    modelo               parametros   tolerado   NAO tolerado
    Wan 2.1 VACE             1,3 B     0,0546        0,0793
    Z-Image v2                ~6 B     0,1421        0,1848
    Krea2 Turbo            12,82 B     0,1199      NAO MEDIDO
    HunyuanVideo 1.5         ~13 B     0,1837        0,2147

Vazia de propósito: `tolerado` registra o maior erro que já se viu **funcionar**, nunca um teto.
Nada acima de 0,1199 foi tentado neste modelo. Esta medição existe para preencher essa célula, ou
para provar que o eixo disponível não alcança.

## O eixo, e por que é o único

`--somente-w4a4` é obrigatório: o caminho W4A8 levanta `convrot rotate kernel only supports
group_size 256` em todo valor que não seja 256 (medido 2026-09-01, `tools/probe_convrot_groupsize.py`),
e o `quant_mixed` mede **os dois** formatos por camada para escolher entre eles — então sem a flag
a conversão recusa antes de começar. Com ela, o W4A4 aceita **16 / 64 / 256 / 1024**, quantiza e
executa nos quatro.

O `--group-size` **não é um eixo aqui**: `quant_mixed.py:360` chama
`ck.quantize_convrot_w4a4_weight(weight, convrot_groupsize, 64)` com o 64 **fixo no código**. Ele
só alcança o caminho W4A8. Lido, não executado — mas é uma constante literal na linha de chamada.

Então o eixo é um só, `convrot_groupsize`, e a direção que aumenta o erro é **para baixo**: cg 64 e
cg 16 contra o cg 256 que já está medido em 0,1199.

## O que o Z-Image mediu neste mesmo eixo (2026-09-03)

    cg     mediana err_w4a4    razao contra cg 256    render
    16           0,1926              1,468x           DESTRUIDO 3/3
    64           0,1516              1,155x           bom
    256          0,1312              1,000x           bom
    1024         menor ainda          < 1             (nao renderizado)

E a hipótese mecânica escrita antes daquela corrida estava **invertida**: previa que grupo maior
daria mais erro ("rotação mais grossa"). Uma rotação de Hadamard de tamanho N espalha cada outlier
por N canais, então N maior mistura **mais** e achata mais o outlier. O controle escrito antes
disparou e impediu a leitura errada.

## Previsões

**P1 — direção.** `err(cg 16) > err(cg 64) > err(cg 256) = 0,1199`, monotônico, medido sobre a
mesma interseção de camadas. Mecanismo: o mesmo do Z-Image acima.
**Refutada se** qualquer par inverter.

**P2 — magnitude.** Aplicando as razões do Z-Image ao 0,1199 do Krea2:
cg 64 entre **0,130 e 0,150**; cg 16 entre **0,160 e 0,195**.
**Refutada se** qualquer uma cair fora da sua faixa. Esta é a previsão fraca do conjunto — ela
assume que a razão entre groupsizes é uma propriedade do formato e não do modelo, e a monotonia
por tamanho já morreu neste mesmo modelo em 2026-09-12.

**P3 — o render, e é aqui que a resposta pode ser "o eixo não alcança".** No cg 16 (~0,176 previsto)
o Krea2 ainda renderiza imagem utilizável em **pelo menos 4 dos 5 prompts**. Razão: 0,176 fica
abaixo dos dois pontos de destruição já medidos nesta bancada, 0,1848 (Z-Image, ~6 B) e 0,2147
(Hunyuan, ~13 B), e o Krea2 tolera 0,1199 com folga visível.
**Se P3 se confirmar, o teto do Krea2 continua NÃO MEDIDO** e o relatório honesto é "tolerado ≥
X, teto acima disso, este eixo não alcança" — não "o Krea2 aguenta qualquer coisa".
**Refutada se** 2 ou mais prompts saírem destruídos no cg 16.

**P4 — o conjunto de camadas.** cg 16 e cg 64 selecionam as **mesmas 224** camadas que o cg 256,
porque `selected_layers` filtra por `shape[1] % convrot_groupsize` e todas as in-features do perfil
`krea2` são múltiplas de 256.
**Refutada se** alguma contagem diferir — e nesse caso a comparação passa a ser só na interseção,
porque comparar medianas sobre conjuntos diferentes é o erro `ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho`.

**P5 — o tamanho dos arquivos é IDÊNTICO nos três.** A escala é `weight_scale` F32 `[rows]`,
**por linha**, não por grupo — conferido no header do `krea2_turbo_w4a4.safetensors`
(`blocks.0.attn.wo.weight_scale F32 [6144]`, 224 tensores I8, arquivo 8.057.499.632 B). O
`convrot_groupsize` muda a rotação, não a contagem de escalas.
**Refutada se** os tamanhos diferirem em um byte.

**P6 — velocidade.** s/passo dos três dentro de **10%** entre si, porque o kernel faz o mesmo
número de MACs.
**Refutada se** algum ficar fora.

## Condições de refutação do experimento inteiro

Qualquer uma destas invalida a corrida, e não o modelo:

1. **O braço de referência não responde.** `avaliar_referencia.py` no BF16 já deu `d_prompt/d_semente
   = 1,7686` nesta mesma máquina; se a corrida do ladder não reproduzir imagem boa no braço BF16,
   nada do resto se lê.
2. **Conjuntos de camadas diferentes comparados como se fossem o mesmo** (ver P4).
3. **Uma corrida só por braço.** O espalhamento entre sementes dentro de um braço chegou a 0,65-0,73
   na corrida de qualidade do Krea2 — maior que as diferenças entre as médias dos braços. Então a
   leitura é **pareada** (mesmo prompt, mesma semente, braços diferentes), nunca a média solta.

## O que esta medição NÃO cobre, em nenhum resultado

Um prompt-set de 5, duas sementes, 10 passos, 1024², euler/simple, cfg 1.0, uma placa, sem métrica
perceptual. Só o `krea2_turbo` (destilado) — o `krea2_raw` não entra. E `cg 1024`, que anda na
direção do **menos** erro, não é medido aqui: esta corrida procura o teto, não o piso.

## Como refazer

    python_embeded\python.exe -s tools\quant_mixed.py --input <bf16> --profile krea2 \
        --calibration calib\krea2_turbo.calib.pt --somente-w4a4 --convrot-groupsize 64 \
        --save-analysis calib\krea2_turbo_cg64.analysis.json --output <...w4a4_cg64.safetensors>

    python_embeded\python.exe -s tools\quality_ladder.py --reference <bf16> \
        --arm <cg256> --arm <cg64> --arm <cg16> --prompts bench\prompts_krea2.txt --seeds 2

---

# RESULTADO -- executado 2026-09-12 na 3090

**Placar: 4 previsões confirmadas, 2 refutadas. E o teto NÃO foi alcançado** -- que é a resposta,
não a falta dela.

## Erro por camada, interseção de 224 camadas nas três (P1, P2, P4)

    cg     mediana      p25      p75      max   razão vs 256   previsto por P2
    256     0,1199   0,0822   0,1475   0,2942        1,000x     (base medida)
     64     0,1238   0,0886   0,1553   0,3397        1,033x     0,130 - 0,150
     16     0,1377   0,1079   0,1961   0,4331        1,149x     0,160 - 0,195

Mesma fonte (`sha 05cab7a3...` nas três análises), mesma calibragem, 224/224 calibradas em todas.

**P1 confirmada.** Monotônica na mediana e em **215 das 224 camadas** individualmente. A camada
que mais sofre é a mesma nos três: `blocks.9.mlp.down`, 0,2354 -> 0,3113 -> 0,4331.

**P4 confirmada.** 224 nas três, interseção completa. Nada foi comparado entre populações
diferentes.

**P2 REFUTADA, e nas duas pontas.** Apliquei ao Krea2 as razões medidas no Z-Image (1,155x e
1,468x) e as duas erraram para o mesmo lado: o Krea2 mede **1,033x e 1,149x**, cerca de **três
vezes menos sensível** ao `convrot_groupsize`. Isso mata a leitura de que a razão entre
groupsizes é propriedade do formato: ela é do modelo, como já era a tolerância. É a **segunda**
hipótese de transferência a morrer neste mesmo checkpoint em um dia -- a primeira foi a
monotonia por tamanho.

## P5: refutada na letra, confirmada no mecanismo

Os arquivos **não** são idênticos: 8.057.499.632 B (cg 256) contra 8.057.499.408 B (cg 64), delta
de **224 bytes**, exatamente um por camada quantizada. Conferido em vez de suposto:

    payload de tensores   8.057.415.984 B nos dois   delta 0
    JSON de metadados     15.719 B contra 15.495 B   delta 224

É `"convrot_groupsize": 256` virando `"convrot_groupsize": 64` -- um caractere a menos, 224 vezes.
O mecanismo que P5 defendia está confirmado: a escala é `weight_scale` F32 `[rows]`, por linha,
então o `convrot_groupsize` não muda quantas escalas existem. **A previsão errou por dizer
"idêntico" onde o certo era "idêntico no payload".**

## A imagem (P3), e por que ela fecha a pergunta pelo lado inesperado

40 renderizações, 4 braços x 5 prompts x 2 sementes. **40 de 40 utilizáveis.** No cg 16 -- o pior
braço, 0,1377 -- a placa "OPEN" sai legível nas duas sementes, o rosto do pescador tem pele e
ruga, o cristal de gelo tem estrutura fina e o mercado noturno é coerente.
`bench/krea2_teto_s1.png` e `_s2.png`, uma folha por semente, BF16 na primeira linha como
controle.

**P3 confirmada, e a consequência é que este eixo não alcança o teto do Krea2.** O
`convrot_groupsize` é a única alavanca que o formato expõe (ver abaixo), o menor valor legal é
16, e no menor valor legal o modelo não quebra.

## O descarte que essa leitura exigia

Imagem boa com erro maior tem uma explicação trivial e errada: o kernel de 4 bits não rodou.
Descartada, executando `probe_quant_dispatch --forward-only` nos dois builds novos:

    cg 64   224 módulos quantizados   8/8 forwards quantizados   0 dequantize   int4   backends.cuda
    cg 16   224 módulos quantizados   8/8 forwards quantizados   0 dequantize   int4   backends.cuda

## Velocidade (P6) -- confirmada, e com uma condição que a enfraquece

    braço       s/passo    GiB
    BF16          2,284  24,48   (descarrega inteiro; não comparável com nada)
    cg 256        0,914   7,50
    cg 64         0,886   7,50
    cg 16         0,972   7,50

0,972 / 0,886 = **1,097x**, dentro dos 10% previstos -- mas por 3 décimos de ponto percentual, e
**parte desta corrida dividiu o disco com um download de 57 GiB que eu mesmo iniciei**. Medi a
taxa de leitura com o download parado (4,8 MB/s) e o ladder não é disk-bound, então o efeito
deve ser pequeno; ainda assim o cg 256 mediu 0,839 na corrida anterior e 0,914 nesta, 9% de
diferença entre corridas sem nenhuma mudança de código. **P6 passa, e um efeito de 10% medido
sob contenção de 9% não é um resultado forte.** Reproduzir em máquina quieta antes de citar.

## A divergência de latente não separa os braços, e isso era esperado

    pareado contra o cg 256, corrida a corrida
    cg 64   +0,0529   pior +0,3282   melhor -0,0215   vence  3/10
    cg 16   +0,0481   pior +0,4578   melhor -0,3037   vence  5/10

A própria ferramenta imprime `Split decisions, i.e. not separated at 10 run(s)`. O braço com
**mais** erro por camada vence metade das corridas pareadas contra o braço com menos. Isso não é
ruído inconveniente: é a mesma coisa que esta bancada já mediu em 2026-08-30 -- a imagem em
trajetória livre mede caos do sampler, não fidelidade. O instrumento certo para separar dois
quants do mesmo modelo é o `probe_epsilon_ckpt_ab`, de entrada casada, e ele continua bloqueado
no Krea2 pela ordem de import da aimdo.

## O que isto muda na tabela de tolerância

    modelo               parâmetros   tolerado   NÃO tolerado
    Krea2 Turbo            12,82 B     0,1377    NÃO MEDIDO -- este eixo não alcança

`tolerado` sobe de 0,1199 para **0,1377**: é o maior erro por camada que já se viu funcionar
neste modelo. A coluna `NÃO tolerado` continua vazia **e agora com motivo escrito**, não por
falta de tentativa.

## Por que não existe um eixo mais forte no caminho suportado

O `quant_group_size` (o outro parâmetro de `quantize_convrot_w4a4_weight`, fixado em 64 pelo
`quant_mixed.py:360`) **não é utilizável como eixo**, e isso foi lido no código, não executado:
`comfy/ops.py:1201` escreve `"quant_group_size": 64` como **constante literal**, enquanto as
duas linhas ao redor leem `convrot_groupsize` (`:1197`) e `linear_dtype` (`:1202`) do JSON da
camada. Um arquivo quantizado com outro valor seria lido pelo loader como 64, e qualquer quebra
seria desacordo entre loader e arquivo -- não tolerância do formato. Medir teto com isso daria um
número sobre um defeito.

Sobra um eixo não tentado: **cobertura**. O perfil `krea2` seleciona 224 Linears e deixa de fora
41 tensores 2-D -- as 32 Linears do `txtfusion`, o `tproj [36864, 6144]`, `tmlp`, `txtmlp`,
`last.linear` e `last.modulation.lin`. Trinta e nove deles passariam o filtro de divisibilidade.
Isso não move a mediana da tabela (muda **quais** camadas são degradadas, não quanto), então
responde a outra pergunta -- "a lista de exclusão do perfil é carregada?" -- e exige recalibrar,
porque o `.calib.pt` atual só tem as ativações das 224. Fica registrado como não feito.

## O que este resultado NÃO cobre

Cinco prompts, duas sementes, 10 passos, 1024², euler/simple, cfg 1.0, uma placa, sem métrica
perceptual, só o `krea2_turbo` destilado. `cg 1024` não foi medido (anda para o lado do menos
erro). O `krea2_raw` não entra. E "40 de 40 utilizáveis" é julgamento de quem olhou as duas
folhas de contato, não uma métrica.
