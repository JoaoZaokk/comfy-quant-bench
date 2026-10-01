# Regras rígidas: formatos, qualidade e critérios

> Referência preservada do CLAUDE.md original, linhas 261–553, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

- **W4A4 means native ConvRot CUDA execution**, not weight-only INT4 followed by BF16 GEMM. Any change that lets the work fall back to eager/dequantized math defeats the entire project.

  **But this rule names two options where the hardware offers three, and the third one wins on accuracy.** Measured 2026-08-30 on the 3090, varying one axis — `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK` (`comfy_kitchen/backends/cuda/__init__.py:212`), which forces the same `convrot_w4a4_linear` down the other branch. `_cuda_device_supports_native_int4_mma` is `major == 8` (`:293`), so Ampere and Ada reach the `m16n8k64 s4` MMA and **Hopper and Blackwell are routed to the INT8 branch deliberately**. The native branch is not merely selected here — it executes: outputs differ from the fallback in 6/6 synthetic cases (`tools/probe_int4_mma_dispatch.py`).

  The third option is **INT4 weight × INT8 activation on tensor cores**, which is neither "native INT4 MMA" nor "dequantized BF16 math". On **real** Z-Image activations and real weights — 24 layers spanning crest 4.4 to 43.0, from `calib/xfer_z_image_turbo_bf16.calib.pt` (`tools/probe_int4_vs_int8_real_acts.py`):

  ```
  media rel-RMSE   nativo 1,28e-1   int8 8,59e-2
  nativo ganha em 0/24 camadas      -> int8 e 1,49x mais fiel
  Spearman(crest, nativo-menos-int8) = +0,247
  ```

  Zero of twenty-four. And this was the run meant to *rescue* the native path: the earlier synthetic measurement used gaussian input, which has no outliers, and outliers are what the rotation exists to suppress — so real activations were expected to narrow the gap. They widened it slightly, 1.4x to 1.49x. Crest does not explain it either (+0.247, weak, and this repo already measured crest against W4A4 error at +0.10).

  Speed is the other half and it points the other way: native is **1.41x and 1.67x faster at M=1024**, and **1.3x to 1.74x slower at M=1** — the `m_crossover` curve shape again.

  So "anything that is not native INT4 defeats the project" is contradicted by measurement, and the two largest public ConvRot distributors — `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` (791k downloads, `w4a4_int4mm_layers: 0`) and `joeygambino/...surgical_int8_convrot` (`target_dtype: int8_tensorwise`) — ship exactly the third option. That reads as a deliberate trade, not a shortcut.

  **Executed 2026-08-31, with the models on the GPU** — `tools/probe_quant_dispatch.py --forward-only` records the device of the weights, the implementation the registry resolved, and the `linear_dtype` that reached the dispatcher, then varies `COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK`:

  ```
  checkpoint                                 linear_dtype  impl     flag muda?  ramo
  zimage-v2-w4a4                 (nosso)         int4      cuda        SIM      nativo int4
  LTX25-distilled-DiT-comfy-w4a4 (riftcast)      int4      cuda        SIM      nativo int4
  MiniMax_H3_FL2VA         (Abiray, 791k dl)     int8      cuda        NAO      INT8, por instrucao
  ```

  The Abiray file **does run quantized math on the CUDA backend**; it is simply insensitive to the int4 fallback flag because those layers never take that branch. Its own summary contradicts itself, too — `convrot_w4a4_mixed` carries `"linear_dtype": "int4"` at the top and `"w4a4_int4mm_layers": 0` three keys below, while all 117 per-layer `comfy_quant` tensors say `"int8"`. Read the layers, not the summary. And `LTX25-distilled-DiT-comfy-w4a4` (`quantized_by: riftcast/ltx25-quant-lab`, 1440 layers, `linear_dtype` absent) is a **second public W4A4 that really executes 4 bits**, and a seventh known checkpoint writer for ticket 08.

  **The rule stands until the owner decides otherwise**, because this is a trade and the choice is his: accuracy and small-batch latency favour the INT8 branch, large-batch throughput favours native.

  **And whether native W4A4 is worth reaching at all depends on the MODEL, which this rule does not mention.** Measured 2026-08-31 on the 3090, requantizing `hunyuanvideo1.5_720p_t2v_fp16` with today's pipeline and re-rendering the exact prompt, seed and sampler of the public repo's August test:

  ```
  modelo             W4A4 por passo   imagem       divergencia
  Z-Image                1.83x-1.93x MAIS RAPIDO   boa      0.7173
  HunyuanVideo 1.5       1.055x MAIS LENTO         DESTRUIDA 0.8255
  ```

  Same format, same kernel, same converter, same `convrot_groupsize` 256, output byte-size identical to August's 7.92 GiB. The public repo's warning — *"ConvRot W4A4 is slower than FP16 and destroys the output"* — **reproduced fifteen days later** across `comfy-kitchen` 0.2.23→0.2.31, ComfyUI 0.29→0.33 and torch 2.12.1→2.13.0. **Nothing in the stack fixed it**, because nothing in the stack was broken: the model is the axis.

  The leading hypothesis before the run was the `to_native.py` fused-qkv gap (scales passing through unrenamed, layer loading with no scale and no error). It died on reading, not on the GPU: `HunyuanVideo.process_unet_state_dict` carries `.comfy_quant` and `.weight_scale` through its own substring replacements — the gap is Z-Image-specific.

  Two consequences. First, **a checkpoint that converts cleanly, resolves the CUDA backend and writes a valid sidecar can still produce garbage** — the preflight proves dispatch, never quality; only a render does. Second, **latent divergence has no threshold**: 0.8255 destroyed against 0.7173 fine is a 15% gap separating "unusable" from "ship it", so no cut on that axis decides anything.

  Not covered: one seed, one prompt, 480x480, one frame, no perceptual metric, no SASS, and only `convrot_groupsize` 256 in this round. Re-run with `tools/quant_w4a4.py --profile hunyuan_video_15` then `tools/quality_ladder.py`; see `W4A4_PROGRESS.md` part 36.

  **A per-layer error threshold was derived from that work, and a third architecture family broke it on 2026-09-01.** The rule read: median `err_w4a4` above 0.21 breaks, below 0.15 works, between the two is correct-but-grainy. Wan 2.1 VACE 1.3B measures **0.1602** — the untested middle — and the render is destroyed, no subject, no bench. Bracketing it with mixed builds puts Wan's line **between 0.0546 and 0.0793**:

  ```
  modelo               parametros   tolerado   NAO tolerado
  Wan 2.1 VACE             1,3 B     0,0546        0,0793
  Z-Image v2                ~6 B     0,1421        0,1848
  Krea2 Turbo             12,82 B    0,1377      NAO ALCANCADO (ver abaixo)
  HunyuanVideo 1.5         ~13 B     0,1837        0,2147  (e 0,2163 no capybara)
  Qwen-Image-Edit 2511    20,43 B    0,0358*       0,1080   (*outro FORMATO, ver abaixo)
  ```

  **A linha do Qwen tem um asterisco porque as duas colunas dela nao sao o mesmo experimento, e
  isso e o achado.** Em toda linha acima, as duas colunas sao W4A4 com groupsize ou limiar de
  promocao diferentes. Na do Qwen, `0,1080` e o W4A4 e `0,0358` e um build **todo em
  `asym_w4a8_int8`** -- mesmos pesos de 4 bits, ativacao de 8 em vez de 4. Medido 2026-09-12, 12
  renderizacoes por braco, 6 prompts x 2 sementes:

  ```
  build                      peso    ativacao   GiB   divergencia  s/passo  imagem
  int8_convrot (Comfy-Org)   8 bits   8 bits   19,09      0,1942    1,410   boa
  w4a8 (nosso)               4 bits   8 bits   10,79      0,4997    1,575   boa
  w4a4 (nosso)               4 bits   4 bits    9,60      1,7440    1,053   ESTATICA
  misto, 607/840 em A4       4 bits   4 bits    9,88      1,8846    1,236   ESTATICA
  ```

  Publicado, com as duas grades como prova -- a do build bom e a dos dois que falharam:
  **https://huggingface.co/JoaoZaokk/Qwen-Image-Edit-2511-W4A8-ConvRot**. Os pesos `w4a4` e
  `misto` NAO subiram: negativo medido se publica como prova, nao como checkpoint que alguem
  baixa e usa.

  **O Wan 2.2 confirmou o eixo numa SEGUNDA familia, com outro modo de falha.** Medido
  2026-09-13, `wan2.2_ti2v_5B`, 6 corridas de 33 quadros a 480px, **os tres bracos RESIDENTES**
  na 3090 -- nenhum descarregado, nenhum espalhado -- entao o s/passo abaixo e comparacao real:

  ```
  braco            GiB   s/passo   divergencia   min-max          imagem
  FP16 original   9,31     1,579        --          --            nitida
  W4A8            2,75     0,994      0,2115   0,1185-0,3275      indistinguivel a olho
  W4A4            2,46     0,753      0,3847   0,2431-0,5748      BORRADA
  ```

  W4A4 e W4A8 tem os MESMOS pesos de 4 bits; so a ativacao difere. No Qwen Edit o W4A4 deu
  estatica, aqui da borrao. **Modo de falha diferente, mesmo eixo, segunda familia.** Publicado:
  **https://huggingface.co/JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot**.

  **E aqui esta o contraexemplo que muda como esta bancada le divergencia de latente.** `0,3847`
  esta DENTRO da faixa que sempre funcionou -- abaixo do Z-Image v2 (0,7854) e do Krea2 (0,5843),
  os dois bons -- e a imagem esta degradada. Ate agora so existia a falha oposta aqui:
  divergencia ALTA com imagem boa, porque uma perturbacao minima manda o sampler para outro lugar
  que tambem presta. Este e o espelho, e o mecanismo e obvio depois de visto: **borrao e uma
  mudanca PEQUENA de latente.** Suavizar afasta menos da referencia do que ir para algo nitido e
  diferente.

  Entao a regra "divergencia de latente nao tem limiar", que este arquivo ja registrava, e fraca
  demais. O certo e: **a divergencia e ENVIESADA para falha macia.** Um build que borra sempre se
  elogia nessa metrica, e o tipo de dano que ela esconde e justamente o mais facil de nao notar
  numa olhada rapida. Nao usar divergencia sozinha para aceitar um build -- nunca foi suficiente,
  e agora se sabe para que lado ela erra.

  Uma armadilha pega antes de custar, no mesmo dia: o latente do `ti2v_5B` tem **48 canais** e o
  `wan_2.1_vae` que mora ao lado tem **16**. Decodificar com o VAE errado da imagem plausivel e
  silenciosamente errada -- o aviso que `decode_latents.py` imprime em toda execucao. Contados os
  canais ANTES, baixado o `wan2.2_vae` (1.409.400.960 B), e so entao decodificado.

  **Nao e o peso, e a ativacao.** E o build misto fecha o mecanismo: 233 camadas ja promovidas a
  ativacao de 8 bits, erro efetivo **0,0744** -- metade do erro de um Z-Image que presta -- e
  ainda assim estatica. **Basta sobrar camada no caminho A4.** E penhasco, nao ladeira.

  **Consequencia para esta tabela inteira: `0,1080` e o MENOR erro por camada que ja produziu
  lixo nesta bancada**, abaixo do Krea2 (0,1199) e do Z-Image (0,1254), ambos funcionando em
  W4A4. O criterio por camada **ordena formatos corretamente** (ele disse que W4A8 era 3,02x
  melhor, e era) e **nao localiza penhascos**. Isso deixou de ser suspeita e virou demonstracao.

  **E toda esta tabela responde uma pergunta de PTQ, que nao e a pergunta que a referencia publica
  de baixissimo bit responde.** Medido 2026-09-21 fazendo engenharia reversa do **Bonsai Image**
  (`prism-ml/bonsai-image-*`, FLUX.2-klein-4B a 1,58 bit) contra o original, peso a peso --
  `bench/bonsai_image_engenharia_reversa.md`, `tools/probe_bonsai_ptq_ou_treino.py`. Tres caminhos
  independentes provam que **eles treinaram**, nao quantizaram: **2.727.589 inversoes de sinal** no
  braco ternario com **0 de 100 camadas limpas** (um PTQ de magnitude nao pode inverter nenhum
  sinal, por construcao); **6,06%** de sinais trocados no braco binario, onde PTQ e *definido* como
  `sign(w)*escala`; e derrota **unanime, 100/100**, para um PTQ ingenuo estilo BitNet na propria
  metrica de distancia ao peso -- o que um quantizador nao pode fazer.

  O mecanismo importa mais que o veredito. O `quantization_config.json` deles declara 100 camadas
  quantizadas e 9 "puladas" por uma lista de nomes escrita a mao, **sem erro medido por camada em
  lugar nenhum**. E o controle revelou por que isso basta: **8 das 9 "puladas" MUDARAM** (rel-L2 ate
  0,235) e **os 60 `norm_q`/`norm_k` tambem**, 0/60 identicos. "Pulada" significa *nao quantizada*,
  nao *nao modificada*: essas camadas ficaram em FP16 para **poderem ser treinadas e compensar** as
  100 esmagadas. O conjunto em alta precisao e **capacidade de adaptacao, nao o conjunto de camadas
  sensiveis**.

  Isso tambem resolve a colisao com a modulacao sem nenhum lado estar errado. Eles pulam os tres
  `*_modulation.linear`; esta bancada mediu `adaLN_modulation` como a camada de **menor** erro em
  W4A4 do bloco (0,1263 contra 0,1569), e isso segue verdadeiro. **Aguentar quantizacao e servir de
  compensador sao propriedades diferentes**, e o mapa deles otimiza a segunda -- por isso a
  modulacao, barata e alimentando 20 blocos, e boa escolha para deixar solta.

  **Entao: o teto do PTQ nao e o teto deles.** Tudo nesta secao -- as tres hipoteses de transferencia
  que morreram, o `--promote-error`, o criterio por camada -- vale para PTQ e nao se compara com um
  modelo que foi treinado na grade. [JULGAMENTO] o proximo ganho real aqui provavelmente nao vem de
  criterio de promocao melhor, e sim de um **estagio de compensacao**: deixar poucas camadas em alta
  precisao e ajusta-las contra a saida do modelo denso, sem retreinar o corpo. O que me faria mudar
  de ideia: se o Bonsai tiver treinado em escala de pre-treino, o mecanismo e "retreinar" e nao cabe
  aqui -- o sinal a favor do ajuste CURTO e que os pesos ternarios ficaram a sinal 0,9999 do original.

  Descartada por medicao a leitura facil de que o arquivo estava quebrado: `probe_quant_dispatch
  --forward-only` da 840 modulos `convrot_w4a4`, **8/8 forwards quantizados, 0 dequantize**,
  `convrot_linear_dtype=int4`, `backends.cuda`, 840 pesos em `cuda:0`; `verify_w4a4` passa em
  estrutura, na comparacao byte a byte com a fonte, e da `relative_rmse` 0,2265 no kernel real.

  **E uma hipotese nova morreu aqui no mesmo dia: erro mediano x numero de camadas.** Ela separa
  NOVE builds perfeitamente (funciona ate 30,9; destruido a partir de 31,4) e morre no decimo,
  que ja estava no disco: `hunyuan15-misto-t025` marca **79,0 e renderiza correto**, contra
  `zimage-v2-teto-cg16` que marca **31,4 e renderiza lixo**. Terceira hipotese de transferencia
  a morrer nesta bancada, depois da monotonia por tamanho e da razao entre groupsizes.

  **A monotonia por tamanho MORREU em 2026-09-12, e ela era o unico argumento para extrapolar
  desta tabela.** O Krea2 Turbo tem **12.820.073.036 parametros** (somados do header, nao
  estimados do tamanho do arquivo) e a previsao escrita antes de medir -- `bench/criterio_quant_krea2.md`
  -- era mediana de `err_w4a4` entre 0,15 e 0,22, porque e ai que o Hunyuan de ~13 B esta. Medido
  sobre 224 camadas, todas calibradas, cg 256: **0,1199**. Um modelo de 12,8 B mede MENOS erro por
  camada que o Z-Image de ~6 B. A linha entra com a coluna `NAO tolerado` vazia de proposito: nada
  foi medido acima de 0,1199 neste modelo, e `tolerado` registra o maior erro que ja se viu
  funcionar, nunca um teto.

  **O teto foi PROCURADO no mesmo dia e o unico eixo suportado nao alcanca.** `bench/criterio_teto_krea2.md`,
  criterio com seis previsoes escrito antes de converter: 4 confirmadas, 2 refutadas. Reconvertendo
  `--somente-w4a4 --uncalibrated fail` nos dois groupsizes menores, medido na **intersecao das
  mesmas 224 camadas** (P4 confirmada, nenhuma populacao diferente comparada):

  ```
  cg     mediana      p25      p75      max   razao vs 256   render (5 prompts x 2 sementes)
  256     0,1199   0,0822   0,1475   0,2942        1,000x     10/10 boas
   64     0,1238   0,0886   0,1553   0,3397        1,033x     10/10 boas
   16     0,1377   0,1079   0,1961   0,4331        1,149x     10/10 boas
  ```

  Monotonico na mediana e em **215 das 224** camadas. **No menor groupsize legal o modelo nao
  quebra**, entao `tolerado` sobe para 0,1377 e a coluna do teto fica `NAO ALCANCADO` -- com motivo,
  nao por falta de tentativa. Descartada a leitura facil e errada de que a imagem sobreviveu porque
  o kernel nao rodou: `probe_quant_dispatch --forward-only` nos dois builds novos da 224 modulos,
  **8/8 forwards quantizados, 0 dequantize**, `convrot_linear_dtype=int4`, `backends.cuda`.

  **A segunda hipotese de transferencia morreu aqui, no mesmo checkpoint e no mesmo dia.** As
  razoes entre groupsizes medidas no Z-Image (1,155x e 1,468x) foram aplicadas ao Krea2 como
  previsao P2; ele mede **1,033x e 1,149x**, tres vezes menos sensivel, e erra para o mesmo lado nas
  duas pontas. A razao entre groupsizes e do MODELO, como ja era a tolerancia -- nao do formato.

  **E `quant_group_size` NAO e um eixo utilizavel, apesar de o parametro existir e o sidecar
  carregar a chave.** `comfy/ops.py:1201` escreve `"quant_group_size": 64` como constante literal,
  enquanto as duas linhas ao redor leem `convrot_groupsize` (`:1197`) e `linear_dtype` (`:1202`) do
  JSON da camada. Quantizar com outro valor produz arquivo que o loader le como 64: qualquer quebra
  seria desacordo loader-vs-arquivo, nao tolerancia do formato. Lido no codigo, nao executado.

  Sobra um eixo nao tentado, a **cobertura**: o perfil `krea2` seleciona 224 Linears e exclui 41
  tensores 2-D -- as 32 do `txtfusion`, `tproj [36864, 6144]`, `tmlp`, `txtmlp`, `last.linear`,
  `last.modulation.lin` -- dos quais 39 passariam o filtro de divisibilidade. Isso nao move a
  mediana (muda QUAIS camadas degradam), entao responde outra pergunta, e exige recalibrar.

  E funciona bem: 5 prompts x 2 sementes x 4 bracos, **40 renderizacoes, nenhuma quebrada** --
  maca (controle), rosto com pele e ruga, placa "OPEN" legivel em 8 de 8 celulas, mercado noturno
  coerente, cristal de gelo com estrutura fina. A divergencia de latente do W4A4 contra o BF16 e
  **0,5843** e a imagem presta: o que esse numero mede e o sampler indo para outro lugar que
  tambem e bom.

  **O int8 do proprio Comfy-Org ganha do nosso W4A4 em 10 de 10 corridas pareadas** (0,2435 contra
  0,5843, 2,40x mais fiel), que e a quarta medicao independente nesta bancada na mesma direcao --
  agora numa quarta familia. E o W4A4 e **1,47x mais rapido por passo** (0,839 contra 1,233) e
  1,68x menor (7,50 contra 12,57 GiB). A escolha e uma troca, nao um erro; o que nao se pode e
  chamar o W4A4 de mais fiel.

  **A celula do Z-Image foi preenchida em 2026-09-03, e a hipotese mecanica que ia preenche-la
  estava invertida.** O eixo e o `convrot_groupsize`: `bench/criterio_teto_zimage.md` previa que
  grupo MAIOR daria mais erro ("rotacao mais grossa"). Medido sobre a intersecao de camadas que
  todos os valores aceitam, quatro pontos monotonicos na direcao **oposta** -- cg 16 `0,1926`,
  cg 64 `0,1516`, cg 256 `0,1312`, cg 1024 menor ainda. Uma rotacao de Hadamard de tamanho N
  espalha cada outlier por N canais, entao N maior mistura MAIS. "Mais grosso" era a intuicao de um
  quantizador por grupo, onde grupo maior significa uma escala para mais valores; a rotacao nao e
  isso. O controle escrito antes (`grupo menor tem de reduzir o erro`) disparou e impediu a leitura
  errada.

  Renderizado com quatro bracos e tres sementes: BF16 bom (controle), cg 256 (0,1216) bom, cg 64
  (0,1421) bom, **cg 16 (0,1848) destruido 3/3**. As tres previsoes escritas antes bateram. Repare
  que a tabela agora e monotonica nas duas colunas e que **0,1848 destroi um modelo de ~6 B
  enquanto 0,1837 e tolerado num de ~13 B** -- 0,6% separando as duas faixas, que e o que se
  esperaria se o tamanho fosse o eixo. Tres pontos continuam sendo tres pontos.

  **E o avaliador offline era cego a este eixo inteiro.** `tools/avaliar.py` casava o
  `.analysis.json` pelo sha da FONTE e lia `err_w4a4` sem olhar o `convrot_groupsize`, entao os
  tres builds -- mesma fonte -- recebiam a **mesma mediana 0,1216** e o mesmo veredito: o que
  desenha bem e o que desenha lixo. Corrigido: camadas com groupsize diferente do da analise sao
  descartadas, e um arquivo sem analise no proprio groupsize ganha o achado
  `analise_de_outro_groupsize` em vez de sair calado. O bloqueio que impedia tudo isso era o
  caminho **W4A8** (so aceita cg 256), nao o ConvRot; `quant_mixed --somente-w4a4` nao mede nem
  escreve W4A8, e em cg 256 produz arquivo **byte a byte identico** ao `zimage-v2-w4a4` publicado.

  Nao coberto: um prompt, tres sementes, um tamanho, uma placa. Nada foi medido entre 0,1421 e
  0,1848, entao o ponto exato da virada nao e um fato -- os fatos sao as duas pontas.

  **A linha do capybara estava na fila errada, e este arquivo a publicou assim.** `capybara_v0.1` foi tratado como checkpoint da familia Z-Image e seu 0,2163 virou o teto do Z-Image. Lido do arquivo em 2026-09-01: **1364 tensores e 54 `double_blocks`** — arquitetura do HunyuanVideo 1.5 — contra os **453 tensores e zero** do Z-Image. Confere tambem por tamanho: 16 653 435 264 bytes contra os 16 653 368 128 do `hunyuanvideo1.5_720p_t2v_fp16`, 67 KiB de diferenca. A quebra e real e passa para a linha de ~13 B, onde concorda com o 0,2147 medido no proprio Hunyuan — e **o teto do Z-Image nunca foi medido**: sabe-se que 0,1241 funciona, e nada alem disso foi tentado. A monotonia na coluna do tolerado sobrevive; uma celula mudou de linha e outra ficou honestamente vazia. Corrigido nos tres cards do HuggingFace e no README publico no mesmo dia.

  Monotone in the tolerated column. So **there is no threshold of the format — there is one per model**, and the practical consequence is that `--promote-error 0.15`, chosen on Z-Image and carried everywhere since, is **not a safe default**: on Wan it writes a file that loads, dispatches natively, passes every structural check, and renders a smear. Three points make the size reading a hypothesis, not a law.

  **And the reference arm broke first, which cost four renders.** The FP16 Wan — no quantization at all — came out as woven fabric at 6 steps/1 frame, at 25/33, at cfg 6 and cfg 1, with and without `ModelSamplingSD3 shift 8` (which applies, and changes no sigma under the `simple` scheduler). The first run's `divergence 1.2365` measured nothing. Cause, one axis varied (`tools/probe_vace_strength.py`): `vace_strength` **1.0** gives `|latent| 607.6` and fabric, **0.0** gives `|latent| 1543.2` and a real workshop. `WAN21_Vace.extra_conds` (`comfy/model_base.py:1710-1737`) fills `vace_frames` with zeros when no VACE node is present, runs each block through `process_latent_in` — which subtracts the latent format's mean, so **zero becomes non-zero** — concatenates an all-ones mask, and applies it at full strength. **Any VACE checkpoint in a plain T2V workflow is destroyed, with no error and no warning.** `quality_ladder.py` now takes `--vace-strength`.

  One counting trap that nearly ended the run early: loading a quantized Wan prints `WARNING: unet unexpected [...comfy_quant]` for all 300 layers. It is **cosmetic** — the tensors are consumed before that check and complained about after. `probe_quant_dispatch.py --forward-only` counts 300 quantized modules, 12/12 quantized forwards, 0 dequantize, native int4 on the CUDA backend.

  Not covered: one prompt, three seeds, one scheduler, one card, no perceptual metric; the checkpoint is **fp16** and is a **VACE variant run as plain T2V**, so the tolerance measured may belong to the mode rather than to the model. See `bench/criterio_wan21.md` (criterion written before the result) and `W4A4_PROGRESS.md` part 43.

  **The "is 1.49x visible?" question is now answered, and answering it corrected how this bench measures.** Three measurements of the same pair, 2026-08-30/31:

  ```
  erro por camada, ativacao real     int8 1,49x mais fiel   24/24 camadas
  epsilon por passo, entrada casada  int8 1,33x mais fiel    8/8 passos
  imagem final, trajetoria livre     2x1 em 3 sementes, ambos a ~0,3-0,5 do BF16
  ```

  The final-image comparison is the one that carries no signal, and it is the one that looked most like an answer. At 8 steps a tiny perturbation reroutes the sampler, and the destination is still a good image — so the free-running image measures chaos, not fidelity. **Do not use a generated image to compare two quantizations of the same model.** `tools/probe_epsilon_per_step.py` is the instrument that does work: the BF16 arm records every `(x, timestep)` it was called with via `model_options["model_function_wrapper"]`, and the quantized arms replay exactly those inputs, so every step is a matched comparison and trajectory divergence cannot exist by construction.

  This also **rehabilitates the per-layer criterion** that `tools/quant_mixed.py` uses. It was written off here on 2026-08-30 as non-predictive; it is not. It predicts the model's prediction error (1.49 against 1.33, same direction, same winner). What it does not predict is the free-running image, and nothing does.

  New information only the per-step view gives: the error is **concentrated at high sigma**. Native goes 6.17e-1 at sigma 1.000 down to 5.31e-2 at sigma 0.300, cosine 0.787 → 0.9986. Quantization damage lands hardest where structure is decided, and decays monotonically into texture. A criterion that weights layers by their contribution at high sigma is therefore a different thing from one that weights all steps equally. **Tested on 2026-08-31, and it does not pay.**

  `calibrate_activations.py` now records the sigma of every sampled row (`sample_sigma`), and `quant_mixed.py` takes `--sigma-weight none|sigma|sigma2|high` (default `none`, unchanged). The weighted criterion is nearly the same criterion: Spearman **+0.9935** (`sigma2`) and **+0.9629** (`high`) against flat on `err_w4a4` over 170 Z-Image layers, moving 6 and 9 layers across the 0.15 threshold.

  **The first comparison said it won 8/8 steps at 1.047x, and that was a confound, not a result.** The weighted build promoted 56 layers to 8-bit against the flat build's 53 — a bigger model, measured against a smaller one. **`ab-so-vale-se-os-dois-tomaram-o-mesmo-caminho` again, with the held axis being the promotion budget.** Always match the budget before comparing two selection criteria.

  **Rebuilt at exactly 56 promotions in all three arms and run over eight seeds, the three criteria are indistinguishable.** Paired design — within a seed the arms share the imposed BF16 trajectory:

  ```
  epsilon medio, 8 sementes    media       min       max    espalhamento entre sementes
  plano56                    1.3326e-1  1.1220e-1 1.5577e-1        1.388x
  sigma2                     1.3684e-1  1.1731e-1 1.5216e-1        1.297x
  sigma_alto                 1.3685e-1  1.2276e-1 1.4695e-1        1.197x

  diferenca pareada contra o plano:  sigma2 +3.37% (erro-padrao 3.95%, vence 4/8)
                                 sigma_alto +3.46% (erro-padrao 3.93%, vence 2/8)
  ```

  The between-seed spread inside a single arm reaches **1.39x**; the between-criteria difference is **3.4%**. The effect sits inside its own noise, and the win counts are coin flips. The sign is consistent — both weighted criteria come out slightly *worse*, never better — but at eight seeds that is a hint, not a result.

  **This corrects the three-seed version of this paragraph**, which reported "flat wins 3/3" and read as a finding. It was a small-sample artifact: adding a third arm made the ranking change between seeds, and the winner across eight seeds is 3 / 3 / 2 split between the three.

  Re-run with `tools/probe_epsilon_ckpt_ab.py`, which imposes the BF16 trajectory on N checkpoints at once. Not covered: one prompt, one model, one scheduler, eight seeds, no perceptual metric, and no formal hypothesis test — only the distance between the effect and its own spread.

  **Not covered:** one prompt, one seed, no perceptual metric, only Z-Image, only sm86, no SASS. And BF16 is the target, not the truth — it was never itself validated against float32.
