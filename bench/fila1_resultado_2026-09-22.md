# Fila 1 na 3090: P1 confirmada a 3,5620x, e o braco 1 bloqueado pelo flux do ComfyUI

Critério em `bench/criterio_fila_gpu_compensacao_2026-09-22.md`, escrito antes de qualquer número.
Lock tomado como `bench:klein4b_compensacao_fila1` com a 3090 em 2 MiB / 0%, e liberado no fim.

---

## 1. P1 CONFIRMADA, e o resultado é melhor que a previsão

`tools/probe_epsilon_ckpt_ab.py`, trajetória do BF16 imposta aos dois braços, 8 passos, 512 px,
semente 1, `cfg 1.0`, euler/simple, encoder Qwen3-4B:

    rel-RMSE do epsilon contra o BF16, entradas casadas
     passo    sigma    braco0_ptq  braco2_bonsai
         0    1.000    1.1268e+00     4.2508e-01
         1    0.981    1.0420e+00     3.6600e-01
         2    0.958    1.0502e+00     3.4793e-01
         3    0.926    1.0586e+00     3.0277e-01
         4    0.883    1.0560e+00     2.6445e-01
         5    0.819    1.0045e+00     2.1812e-01
         6    0.715    8.9752e-01     1.7164e-01
         7    0.519    6.5529e-01     1.1935e-01

    faixa                 braco0_ptq  braco2_bonsai   quem ganha
    todos os passos        9.8637e-01     2.7692e-01   Bonsai 3,5620x mais fiel
    sigma ALTO (>= 0.926)  1.0694e+00     3.6044e-01   Bonsai 2,9669x
    sigma BAIXO (< 0.926)  9.0333e-01     1.9339e-01   Bonsai 4,6710x
    passos vencidos                 0              8   de 8

Previsão: epsilon do braço 0 **>= 3x** o do braço 2, refutando abaixo de 1,5x. **Deu 3,5620x e 8/8.**

**E a ordem INVERTE entre os dois espaços, que é o achado:**

    espaço de PESO    braço 0 rel-L2 0,46626   Bonsai 0,53474   -> braço 0 MAIS PERTO
    espaço de SAÍDA   braço 0 rel-RMSE 0,986   Bonsai 0,277     -> braço 0 3,5620x PIOR

O PTQ ingênuo fica a rel-RMSE **~1,0**: a previsão dele está tão longe do BF16 quanto o próprio
sinal. Isso é a demonstração mais direta que esta bancada consegue de que **o treino do Bonsai
comprou algo que distância de peso não vê** — e é o mesmo padrão que este repo já registra para erro
por camada, que ordena formatos e não localiza penhasco.

**P4 REFUTADA na direção prevista.** Eu previ o ganho concentrado em sigma alto; ele é **maior em
sigma baixo** (4,6710x contra 2,9669x). Os dois braços melhoram quando sigma cai, e o Bonsai melhora
mais rápido.

## 2. O braço 1 está BLOQUEADO, e o bloqueio não é de recurso

`tools/ajusta_denso_compensacao.py` chegou a funcionar até o backward: professor gravado (8 exemplos),
aluno carregado, **9 tensores treináveis com 195.035.136 valores**, otimizador montado. O backward
morre:

    RuntimeError: one of the variables needed for gradient computation has been modified by an
    inplace operation: [CUDABFloat16Type [1, 4608, 3072]], which is output 0 of Add,
    is at version 20; expected version 19 instead

A causa está lida no código, em cinco linhas:

    comfy/ldm/flux/layers.py:246   img += apply_mod(self.img_attn.proj(img_attn), ...)
    comfy/ldm/flux/layers.py:248   img += apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), ...)))
    comfy/ldm/flux/layers.py:251   txt += ...
    comfy/ldm/flux/layers.py:253   txt += ...
    comfy/ldm/flux/layers.py:352   x   += apply_mod(output, mod.gate, ...)

`self.img_norm2(img)` guarda `img` para o próprio backward e a linha seguinte o muta no lugar. **A
implementação do flux no ComfyUI é escrita para inferência e não é diferenciável.**

**E não é questão de versão: o upstream v0.37.0-11 tem as MESMAS cinco linhas, in-place, nos mesmos
números.** Lido na árvore extraída, não suposto.

Existe um caminho de treino no ComfyUI e ele **não** é suficiente:
`comfy.model_management.in_training` troca o RoPE fundido do comfy_kitchen pela versão pura
(`comfy/ldm/flux/math.py:47-51`) e resolve o `Trying to backward through
comfy_kitchen.apply_rope.default but no autograd formula was registered` — mas não toca as somas
in-place. Ligado, medido, e o erro seguinte é o de cima.

### As duas saídas, e as duas são decisão dele

1. **Tornar as 5 somas out-of-place no core** — um sexto patch local em `ComfyUI/comfy/ldm/flux/`,
   no arquivo que todo render de flux atravessa. Cinco linhas, com um controle de igualdade byte a
   byte sob `no_grad` provando que o forward corrigido não mudou nada. É pequeno, mas **aumenta a
   dívida de patch que eu acabei de documentar como o custo de atualizar o ComfyUI**.
2. **Reimplementar os dois `forward` por monkeypatch só na ferramenta** — não toca a instalação, mas
   é reimplementar ~60 linhas de bloco, e uma diferença silenciosa envenenaria o experimento. Exige
   o mesmo controle de igualdade.

Não escolhi nenhuma das duas sozinho: as duas mudam como o modelo calcula, e a primeira mexe no core.

## 3. Cinco defeitos meus nesta sessão, e um deles é de processo

1. **Os dois probes de epsilon nunca carregavam o `extra_model_paths.yaml`** — terceiro tool da
   bancada com esse buraco. Consertado, e o conserto quebrou o tool **duas vezes**: um printf no
   template virou especificador de `%`-format, e o comentário em que eu explicava isso continha o
   próprio caractere. Uma terceira tentativa gerou o patch por script e os escapes saíram mangled.
   Escrito direto na quarta, com `injeta_yaml` levantando exceção se a marca de boot faltar.
2. **Li um log velho como se fosse novo.** O `ruff` falhou com `PERF102` e o `&&` impediu o script de
   rodar; os números vinham byte a byte idênticos porque eram a corrida anterior, e as minhas
   instrumentações novas não apareciam. **O meu próprio encadeamento mascarou a falha.** Agora apago
   o log antes e não encadeio lint com execução.
3. **Diagnostiquei o OOM como sendo no backward. Era no forward do professor.** O gradient
   checkpointing que eu adicionei tratava um problema que não estava ali.
4. **Depois atribuí o OOM à ordem de carga.** Aquilo era um defeito real e o conserto é bom — o
   encoder de 8,04 GiB e o difusor de 7,2 não cabem juntos, e agora o encoder sai antes — mas **não
   era a causa daquele OOM.**
5. **A causa real era eu passar o lado em PIXELS como lado do LATENTE.** `torch.zeros([1, C, 512,
   512])` em vez de `64, 64`: **64x tokens a mais**, e os 4,50 GiB que o alocador pedia. O probe que
   funciona faz `SIDE // 8` e estava ali para ser lido. Três diagnósticos errados antes de olhar o
   número: 4,50 GiB para um MLP são ~244 mil tokens, e isso sozinho dizia onde procurar.

Também medido de passagem: **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` não funciona nesta
plataforma** (`expandable_segments not supported on this platform`), então aquela sugestão do próprio
torch é inútil no Windows.

## 4. Correção ao critério: o conjunto denso em nomenclatura BFL são 9 tensores, não 69

O critério diz "69 tensores, 195.042.816 parâmetros". Em nomenclatura BFL, as 60 normas do klein
ficam **dentro** dos blocos (`double_blocks.N.img_attn.norm.key_norm.scale`), então a regra
estrutural seleciona **9 tensores, 195.035.136 valores**. A diferença em parâmetros é de 7.680 —
exatamente as 60 normas de 128 — mas a contagem de tensores muda e o critério fica corrigido aqui, não
lá.

## Não coberto

- **Uma semente, um prompt, uma placa, 8 passos, 512 px, sem métrica perceptual.** A P3 pede 8
  sementes e isso **não** foi feito: com um efeito de 3,56x contra uma linha de refutação em 1,5x a
  margem é grande, mas o número de uma semente não carrega a precisão que quatro dígitos sugerem.
- **Nenhuma imagem gerada.** Isto mede a previsão do modelo em entradas casadas.
- **O braço 1 não existe**, então P2 (fechar >= 25% da distância) não foi testada, e os dois
  controles obrigatórios (`zero` byte a byte, `ruido` tem de piorar) não rodaram.
- O BF16 é o alvo, não a verdade: nunca foi validado contra float32.
- Nenhum número de tempo: o corpo ternário roda desempacotado em bf16, sem kernel de 1,58 bit.
