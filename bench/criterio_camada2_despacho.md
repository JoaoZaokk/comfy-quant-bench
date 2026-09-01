# Criterio da camada 2 do avaliador: "a matematica quantizada executa HOJE?"

Escrito em 2026-09-01 **antes** de rodar qualquer checkpoint, para que o criterio nao seja
inventado depois de ver o resultado.

## A pergunta, e por que a camada 1 nao pode responde-la

A camada 1 (`tools/avaliar.py`) le cabecalho, sidecar e analise. Ela ve o campo `backend` do
`.quant.json` e sabe que os kernels resolveram para `comfy_kitchen.backends.cuda` -- **na hora da
conversao**. O proprio cego dessa checagem ja diz: *"e o registro de uma conversao passada, nao
uma execucao agora"*. Entre aquela conversao e hoje mudaram comfy-kitchen (0.2.23 -> 0.2.31),
ComfyUI (0.29 -> 0.33) e torch (2.12.1 -> 2.13.0), e nenhum deles avisa quando um formato deixa de
resolver.

A camada 2 carrega o modelo pelo caminho normal do ComfyUI, roda um forward e **conta**.

## O instrumento ja existe

`tools/probe_quant_dispatch.py` faz exatamente essa contagem para UM checkpoint. A camada 2 e o
lote: percorre os checkpoints que a camada 1 marcou como tendo camadas quantizadas (37 de 161
laudos, medido hoje), chama o probe por subprocesso, e resume.

Nao reimplementar a contagem. O probe ja resolveu os dois erros dificeis -- ler o kwarg que o
proprio `ops.py` calcula em vez de re-derivar `_use_quantized`, e instrumentar **depois** do load,
porque o load dequantiza de forma legitima.

## O conjunto de vereditos, e por que `APROVADO` continua ausente

| veredito | significa |
|---|---|
| `DESPACHA` | rodou, e toda Linear quantizada fez matematica quantizada |
| `MISTO` | rodou, e parte fez, parte nao. Ler camada a camada |
| `NAO_DESPACHA` | rodou, e **zero** forwards quantizados: economia de memoria, nao de tempo |
| `TRAVADO_PELO_COMFY` | zero forwards quantizados **por decisao do ComfyUI**, nao do arquivo |
| `NAO_CHEGOU_QUANTIZADO` | a camada 1 leu camadas quantizadas no arquivo e o loader nao reconheceu **nenhuma** |
| `SEM VEREDITO` | o forward nao completou, ou nao carregou. Os contadores nao dizem nada |
| `NAO_PROBAVEL` | o arquivo nao esta em `diffusion_models/` nem `text_encoders/`, entao o probe nao o resolve |

As duas ultimas linhas foram acrescentadas depois de ler o codigo do probe e contar as pastas, e
**antes de rodar qualquer checkpoint** -- sao caminhos que existem no codigo, nao reacoes a
resultado. `NAO_CHEGOU_QUANTIZADO` e o `rep["erro"]` do probe; medido hoje, 2 dos 37 alvos vivem em
`checkpoints/` e caem em `NAO_PROBAVEL`. Registrar as duas em vez de silencia-las importa porque
`NAO_PROBAVEL` e uma limitacao da ferramenta e `NAO_CHEGOU_QUANTIZADO` e um achado sobre o arquivo,
e as duas produziriam zero forwards quantizados se fossem somadas.

`DESPACHA` **nao e aprovacao** e nao deve ser lido como tal. Ele responde uma pergunta binaria e
decidivel -- o kernel foi chamado? -- e nada mais. Um checkpoint que despacha nativamente pode
gerar lixo: medido nesta bancada, o HunyuanVideo 1.5 W4A4 despacha e a imagem e destruida. Por
isso o conjunto tem o nome do FATO (`DESPACHA`) e nao do julgamento (`APROVADO`), que continua
ausente do avaliador inteiro.

## A armadilha que decide se esta camada presta: o text encoder e travado de fabrica

`comfy/sd.py:269` chama `set_model_compute_dtype(torch.float32)` para **todo** objeto CLIP, o que
liga `comfy_force_cast_weights` em cada modulo e faz a matematica cair para dequantizada. Medido em
2026-08-31: um encoder ConvRot INT4 publico de 13,2 GiB roda **350 dequantize e zero** chamadas ao
caminho de 4 bits, e o nosso `gemma_3_12B_it_heretic_w4a8` da 0 quantizados contra 336 dequantize.

Isso **nao e defeito do checkpoint**. Uma camada 2 que reportasse `NAO_DESPACHA` para todo text
encoder estaria certa nos numeros e errada na conclusao, e a tabela mandaria alguem reconverter
arquivos que estao bons. Entao: quando `mode == te` e a trava esta ligada em todas as camadas, o
veredito e `TRAVADO_PELO_COMFY`, com a causa nomeada na mensagem.

**Correcao aplicada depois de rodar, e esta e a versao que vale:** sao DUAS travas, nao uma. Alem
do `comfy_force_cast_weights` acima, ha o `full_precision_mm`, hardcodado em `comfy/sd1_clip.py:114`
para todo text encoder. A primeira versao desta regra pedia so a primeira e por isso classificou
mal justamente o arquivo que estabeleceu o achado. Detalhe e numeros na secao RESULTADO, no fim.

## A outra armadilha: zero significa duas coisas opostas

Contador zerado por **nao ter executado** e contador zerado por **ter executado dequantizado** dao
o mesmo numero e significam coisas contrarias. O probe ja separa isso pelo campo `rodou`, e a
camada 2 tem de propagar a separacao em vez de somar. Um forward que morreu vira `SEM VEREDITO`,
nunca `NAO_DESPACHA`.

## Previsoes, escritas antes de rodar

| checkpoint | previsao | por que |
|---|---|---|
| `zimage-v2-w4a4` (difusao, 170 convrot) | `DESPACHA` | **controle positivo**: medido 2026-08-31, 340 quantizados / 0 dequantize |
| `gemma_3_12B_it_heretic_w4a8` (TE) | `TRAVADO_PELO_COMFY` | **controle negativo**: medido 2026-08-31, 0 / 336 dequantize |
| `LTX25-distilled-DiT-comfy-w4a4` (1440 camadas) | `DESPACHA` | nunca carregado aqui; o dialeto por tensor diz int4 real |
| `ltx-2.3-22b-dev-fp8` (1496 float8_e4m3fn) | nao sei | fp8 e outro caminho de dispatch; sem medicao anterior |
| `MiniMax_H3_*` (Abiray) | `SEM VEREDITO` | o forward quer uma LISTA de latentes (video+audio) e morre em `audio_src = x[1]` |

As duas primeiras sao verdade conhecida e existem para poder derrubar a ferramenta. As tres
ultimas sao previsoes de verdade -- podem errar sem que isso invalide a camada.

## Condicoes de refutacao (se qualquer uma acontecer, NAO publicar a tabela)

1. **O controle positivo falha.** Se `zimage-v2-w4a4` sair diferente de `DESPACHA`, a ferramenta
   nao esta medindo despacho, e nenhuma outra linha vale nada.
2. **O controle negativo passa.** Se `gemma_3_12B_it_heretic_w4a8` sair `DESPACHA`, a camada nao
   distingue kernel de dequantizacao -- que e a unica coisa que ela existe para distinguir.
3. **Um forward que falhou recebe veredito.** Se qualquer linha com `rodou=False` sair diferente de
   `SEM VEREDITO`, a distincao entre "zero porque nao rodou" e "zero porque dequantizou" se perdeu,
   e a tabela inteira passa a misturar as duas.

## O que esta camada NAO cobre, e vai impresso em toda execucao

- **Nada sobre qualidade.** Despachar nativamente e ortogonal a gerar imagem boa. Medido aqui:
  HunyuanVideo 1.5 W4A4 despacha e o render e destruido; Wan 2.1 VACE misto despacha e o render e
  uma mancha. Para qualidade, camada 3 (`avaliar_referencia.py`) mais alguem olhando.
- **Nada sobre fidelidade.** Nenhuma referencia BF16 casada e carregada, entao nenhum numero aqui
  compara contra nada.
- **Poucos passos, lado pequeno.** O bastante para o dispatch acontecer, nao para julgar imagem.
- **Uma placa, um build.** comfy-kitchen 0.2.31, ComfyUI 0.33.0, torch 2.13.0+cu130, sm86. O
  veredito e sobre ESTA maquina hoje, que e precisamente o ponto da camada -- e tambem o limite
  dela.
- **Sem SASS.** "Chamou o kernel" vem de contador em Python, nao de instrucao de maquina.
- **Um checkpoint que a camada 1 nao marcou como quantizado nao e visitado.** Ausencia de linha
  nao e aprovacao nem reprova.

---

# RESULTADO -- executado 2026-09-01 na 3090

Criterio acima escrito antes. Nenhum limiar nem veredito foi mexido depois de ver numero; as duas
correcoes registradas abaixo sao de LOGICA errada, apanhadas por evidencia, e estao descritas.

## As tres condicoes de refutacao ficaram caladas

    controle positivo   zimage-v2-w4a4              esperado DESPACHA            veio DESPACHA
    controle negativo   gemma_3_12B_it_heretic_w4a8 esperado TRAVADO_PELO_COMFY  veio TRAVADO
    forward que falhou  os 2 MiniMax do Abiray      SEM VEREDITO, nao NAO_DESPACHA

## 37 alvos

    26  DESPACHA              todo checkpoint de difusao desta bancada, 0 dequantize
     6  TRAVADO_PELO_COMFY    os seis text encoders
     2  SEM VEREDITO          MiniMax_H3_{FL2VA,Ref2VA} do Abiray -- morrem em 7-9 s
     2  NAO_PROBAVEL          vivem em `checkpoints/`
     1  NAO_DESPACHA          flux-2-klein-base-4b-fp8

Os 26 incluem **treze builds nunca carregados aqui antes**: os quatro `zimage-v2-mixed-t0.*`, os
tres `zimage-v2-sigma-*`, os cinco `hunyuan15-misto-t*` e os tres Wan. Tambem passam tres arquivos
de terceiros -- os `LTX25-distilled-DiT-comfy-*` do riftcast (1440 camadas cada), o
`DasiwaWAN22I2V14BLightspeed` (400 `int8_tensorwise`) e o `minimax_h3_..._w4a8_convrot` do
Winnougan. Entao o campo `backend` do sidecar, que so registra a conversao, **continua descrevendo
a execucao de hoje** -- que era exatamente a duvida que abriu esta camada.

## Previsoes: 4 de 5

| checkpoint | previsto | medido |
|---|---|---|
| `zimage-v2-w4a4` | DESPACHA | DESPACHA |
| `gemma_3_12B_it_heretic_w4a8` | TRAVADO | TRAVADO |
| `LTX25-distilled-DiT-comfy-w4a4` | DESPACHA | DESPACHA |
| `MiniMax_H3_*` (Abiray) | SEM VEREDITO | SEM VEREDITO |
| `ltx-2.3-22b-dev-fp8` | "nao sei" | NAO_PROBAVEL -- vive em `checkpoints/`, nem chegou a carregar |

A ultima nao e acerto nem erro: a pergunta continua aberta, e o `flux-2-klein-base-4b-fp8` a
respondeu por outro caminho (abaixo).

## Dois defeitos da propria ferramenta, apanhados por evidencia

**Sao DUAS travas de text encoder, e a regra so conhecia uma.** O `CLAUDE.md` documenta as duas --
`comfy_force_cast_weights` (de `comfy/sd.py:269`) e `full_precision_mm` (hardcodado em
`comfy/sd1_clip.py:114` para todo text encoder) -- e a regra pedia so a primeira. Cinco encoders
tem as duas; o sexto nao:

    5 encoders                            force_cast {'True': N}     -> TRAVADO
    qwen3vl_32b_minimax_h3-int4_convrot   force_cast {'False': 351}
                                          fpmm       {'True': 350}   -> caia em NAO_DESPACHA

Esse sexto e **o mesmo arquivo com que esta bancada ESTABELECEU a trava dos encoders**. Ele saia
`NAO_DESPACHA`, que le como defeito do checkpoint e mandaria alguem reconverter um arquivo publico
que esta bom. Basta uma das duas para a matematica cair, entao a regra passou a pedir uma das duas
e a NOMEAR qual, porque as duas tem origem e conserto diferentes.

**O laudo nao gravava a evidencia do proprio veredito.** `TRAVADO_PELO_COMFY` depende inteiramente
de `comfy_force_cast_weights`, e o JSON gravado nao trazia esse campo -- dava para ler
`NAO_DESPACHA` e nao ter como conferir por que nao foi `TRAVADO`. Um veredito cuja evidencia nao
esta no relatorio e uma opiniao. As duas travas passaram a ser gravadas e as seis linhas afetadas
foram remedidas.

Note que o segundo defeito e o que permitiu achar o primeiro: enquanto o campo nao era gravado, a
unica coisa visivel era um veredito plausivel.

## O achado: fp8 roda dequantizado, e o contador de `impl` diz na cara

    flux-2-klein-base-4b-fp8   difusao   78 camadas float8_e4m3fn
      forwards quantizados  0
      dequantize            8
      impl                  dequantize_per_tensor_fp8=comfy_kitchen.backends.cuda  x8
      force_cast            {'False': 78}      <- NAO e a trava do CLIP
      full_precision_mm     {'True': 78}

E modelo de difusao, nao encoder, e `comfy_force_cast_weights` esta False -- entao nao e a trava
que prende os text encoders. A operacao literalmente chamada e a de **dequantizar**. Economia de
VRAM, nao de tempo.

**Aberto, e nao vou adivinhar:** de onde vem `full_precision_mm=True` num modelo de difusao. O
`CLAUDE.md` registra que `comfy/ops.py:1667` passa `disabled=` e nao `full_precision_mm` nesse
caminho. Uma amostra, um checkpoint fp8, uma placa.

## Nao coberto, alem do que ja estava escrito acima

- `--forward-only` chama os modulos REAIS que o loader produziu, com entrada sintetica da forma
  certa, e so os **8 primeiros**. Prova que aqueles modulos, como o loader os deixou, despacham --
  nao que uma geracao inteira dispare. As contagens de trava, essas sim, cobrem todos os modulos.
- Os dois `NAO_PROBAVEL` e os dois `SEM VEREDITO` continuam sem resposta. Ausencia de veredito nao
  e aprovacao nem reprova, e sao 4 dos 37.
- O `impl:` conta o que o registry RESOLVEU, nao o que executou. No `qwen3vl` ele marca
  `convrot_w4a4_linear=...cuda: 7` com zero forwards quantizados e zero dequantize -- resolucao e
  execucao sao coisas diferentes e a linha nao deve ser lida como kernel rodando.
