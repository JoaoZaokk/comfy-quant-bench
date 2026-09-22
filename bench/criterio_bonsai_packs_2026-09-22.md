# Criterio: os tres packs do Bonsai Image que eu NAO tinha aberto

Escrito **2026-09-22, antes de qualquer numero**, enquanto o download ainda roda. Fecha o item
"o pack binario (INT1) e os packs MLX nao foram abertos" do `bonsai_image_engenharia_reversa.md`.

O que ja estava aberto (e de onde vem tudo que ja afirmei):

    bonsai-image-ternary-4B-unpacked        medido  -- 2.727.589 inversoes de sinal
    bonsai-image-binary-4B-unpacked         medido  -- 6,06% de sinais trocados
    bonsai-image-ternary-4B-gemlite-2bit    medido  -- 2,5000 bits/peso, grupo 128 em K
    FLUX.2-klein-4B                         medido  -- o original

O que abre agora, so os transformers (text encoder e VAE ficam de fora de proposito: ja conferidos
por tamanho contra os unpacked, e sao 2,1 GB por repo que nao respondem nada aqui):

    bonsai-image-binary-4B-gemlite-1bit/transformer-gemlite-int1/state_dict.pt      1030,34 MiB
    bonsai-image-ternary-4B-mlx-2bit/transformer-packed-mflux/...safetensors        1359,24 MiB
    bonsai-image-binary-4B-mlx-1bit/transformer-packed-mflux/...safetensors          920,49 MiB

---

## As perguntas, e por que valem a pena

**A de verdade e a P2/P3, e a premissa delas precisa de uma correcao que faco ANTES de ver numero:**
eu nunca desempacotei os bits do gemlite. O `probe_bonsai_pack_gemlite.py` e um dumper de
ESTRUTURA -- nome, dtype, bytes por tensor -- e o `2,5000 bits/peso` saiu dessa contabilidade
(slot + duas tabelas fp32 por grupo), nao de ler slot nenhum. Toda a comparacao de codigo que
sustenta o veredito de treino foi feita no repo **unpacked**, que publica tensor comum.

Isso e uma boa noticia para o veredito e muda o que o MLX pode testar. Ele nao valida um
desempacotamento meu que nao existe. O que ele testa e outra coisa, ainda util: **as tres
distribuicoes carregam os MESMOS pesos treinados?** Se o MLX desempacotado bater com o repo
unpacked, as tres sao vistas do mesmo modelo. Se nao bater, alguem requantizou por pack e a
frase "mesmos pesos, empacotadores diferentes" cai.

---

## Previsoes refutaveis

**P1 — o pack binario gemlite da 1,5000 bits/peso.**
Modelo: mesmo formato do ternario, so o slot encolhe. Slot de 1 bit + duas tabelas fp32 por grupo de
128 (escala e zero) = `1,0 + 32/128 + 32/128` = **1,5000**.
*Refuta se* der outro valor. O alternativo interessante e **1,2500**: seria o formato largando a
tabela de zero no caso binario (um codigo de dois niveis nao precisa de zero-point), o que
significaria que o gemlite especializa por largura em vez de ter um layout unico -- e enfraqueceria
a generalidade da minha correcao dos 2,5000.

**P2 — o transformer MLX ternario, desempacotado, bate com `bonsai-image-ternary-4B-unpacked`:
concordancia de codigo 1,000000.**
O unpacked e a referencia porque e o unico que publica tensor comum, sem layout a adivinhar.
*Refuta se* discordarem materialmente -- ai alguem requantizou por pack, e "mesmos pesos,
empacotadores diferentes" cai.

**P3 — o mesmo para o binario: MLX contra `bonsai-image-binary-4B-unpacked`, 1,000000.**

**P4 — os packs MLX tambem agrupam 128 no eixo K.**
*Refuta se* o numero de grupos por linha nao der `K/128`.

**P5 — o codigo binario tem ZERO zeros.**
"Binario" so faz sentido como dois niveis. Se houver posicao zero, o nome esta errado e e ternario
esparso disfarcado.
*Refuta se* a fracao de zeros for diferente de 0,000.

**P6 — os 69 tensores densos bf16 aparecem identicos nos packs MLX.**
Eles sao as camadas puladas + normas, que ja medi como *modificadas* em relacao ao original mas que
devem ser as MESMAS entre packs do mesmo modelo.
*Refuta se* diferirem entre packs.

---

## O controle que TEM de falhar

Comparar os codigos **ternarios do MLX** contra o **unpacked BINARIO**. Sao modelos diferentes
(treinos diferentes, larguras diferentes) e **tem de discordar forte**. Se a minha comparacao disser
que batem, a comparacao esta quebrada e nada de P2/P3 vale.

---

## Desfecho que nao e fracasso

**"Nao consegui desempacotar o formato mflux" e um resultado legitimo** e vai escrito assim, com o
que eu tentei. O que nao pode acontecer e eu inventar um layout que faz os numeros baterem: se eu
tiver de escolher entre dois layouts plausiveis pelo resultado que eles produzem, **isso e ajuste ao
alvo** e invalida P2/P3 inteiras. O layout sai do config/metadata deles ou de nada.

## O que isto NAO responde, aconteca o que acontecer

- Nenhuma imagem, nenhuma ativacao, nenhuma velocidade. Segue tudo em espaco de peso.
- Nao diz se o Bonsai presta. Diz se as empacotadoras concordam e quanto custa o slot.
- Nao toca a GPU.
