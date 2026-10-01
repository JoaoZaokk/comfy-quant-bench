# Recado TTS, publicação e backward

> Referência preservada do CLAUDE.md original, linhas 1–85, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## RECADO TEMPORARIO — do projeto de TTS/STT (2026-09-18). Apagar esta secao depois de agir.

Escrito pela sessao que roda em `P:\PROJETOS\TTS_TRAINING_LORA` a pedido do dono. Sao tres coisas
que sairam de la' e tocam este repo. **Verificar, agir, e remover este bloco inteiro.**

### 1. Risco latente: `.hf/` nao esta no `.gitignore`

MEDIDO aqui em 2026-09-18. **Nada vazou** — o risco e' prospectivo, nao retroativo. Conferido nos
**260 commits** do historico, por quatro caminhos independentes:

    objetos cujo path contem `.hf/`, em todo o historico      0
    `git log --full-history -- .hf .hf/*`                     vazio
    token literal `hf_` + 30 ou mais caracteres, 260 commits  nenhum
    `sk-`, `ghp_`, `github_pat_`, `AKIA`, chave PEM           nenhum

(Um `git log -S "hf_"` devolve ~10 commits, mas e' `hf_parallel_get.py` e `hf_download` — nome de
arquivo e de funcao, nao token. Por isso a busca com comprimento minimo, que separa os dois.)

**Nao ha' nada a revogar nem historico a reescrever.** O que existe e' que `.hf` **nao aparece no
`.gitignore`**, enquanto o `W4A4_HANDOFF.md:611` documenta por escrito que o token de ESCRITA do
HuggingFace vive em `.hf/token`. Hoje o arquivo esta' fora do repo porque ninguem o adicionou, nao
porque alguma regra impeca — um `git add -A` distraido bastaria.

**FEITO em 2026-09-20.** O dono concordou e `.hf/` esta no `.gitignore:119`, conferido por prova
positiva (`git check-ignore -v .hf/token` responde `.gitignore:119:.hf/`). A porta fechou antes de
ser usada, e a conclusao do levantamento acima -- nada vazou nos 260 commits -- nao mudou.

**E no mesmo dia essa mesma pasta cobrou o preco por outro lado:** `HfApi()` sem token explicito
pega o que o ambiente tiver, e `F:/hf-cache/token` e de **leitura**. O upload comeca e morre em
`403 Forbidden: you must use a write token` **depois** de a transferencia ja estar andando. Os dois
arquivos tem **37 bytes**, entao tamanho nao distingue qual e qual. Quem sobe qualquer coisa daqui
fixa `HF_HOME = <repo>/.hf` primeiro -- `.scratch/sobe_fecha.py` agora faz isso a partir de
`__file__`, e a permissao se confere por **prova positiva** (subir 1 byte e apagar), nunca por
`whoami`, que responde igual para token de leitura.

### 2. Ferramenta: varredor de vazamento antes do push

`github.com/JoaoZaokk/speech-quant-lab` -> `bench/varrer_antes_de_publicar.py`. Procura caminho de
disco, compartilhamento de rede, home de usuario, e-mail, chave/token e caminho de dado de saida.
Sai com codigo 1, entao cabe num hook de pre-commit.

Rodando aqui ele acusa **1708 ocorrencias**, e a maioria **nao e' defeito** — este repo publica
handoffs que citam `F:\COMFY_PORTABLE` de proposito, porque sao instrucao para o agente. Se for
usar, alimente `.publicacao-isencoes` com esses casos e o relatorio fica util; sem isso vira ruido,
que e' pior que nao ter relatorio porque da' a sensacao de ter conferido.

### 3. Achado tecnico que toca o `compile_support.py` deste repo

Este repo registra `convrot_w4a4_linear` como `torch.library.custom_op` porque o Dynamo tracava
para dentro do kernel. Ha' uma segunda armadilha na mesma familia, e ela NAO aparece como erro —
aparece como ganho que some:

**Escrever o backward como `torch.autograd.Function` em Python anula o ganho sob `torch.compile`.**
O Dynamo nao consegue tracar o `backward()` dela (e' Python arbitrario), embrulha em
`autograd_function_apply` e marca o ponto como opaco; o AOTAutograd entao nao gera o backward dentro
do grafo e **o backward inteiro volta a ser eager**. MEDIDO no S2 Pro: com DOIS lineares trocados o
estrago ja' era de 75 ms num passo de 183 ms — nao e' custo por chamada, e' o grafo desmontando.

O conserto e' `torch.library.register_autograd` no op, e ai' o backward vira um op comum que o
AOTAutograd poe no grafo. So' importa se algo aqui fizer BACKWARD (treino/LoRA); para inferencia
pura nao muda nada.

De quebra, dois numeros que podem servir de referencia, medidos numa 3090 sm86 com
`comfy_kitchen::int8_linear` (cuBLASLt, peso ja' em int8, ativacao quantizada por linha dentro do
kernel), M = 145 linhas:

    GEMM              bf16    int8    _int_mm + sanduiche
    155776 x 2560    2,302   0,892    3,844     2,58x  contra  0,60x
    9728 x 2560      0,220   0,119    0,339     1,85x
    2560 x 9728      0,201   0,102    0,500     1,98x

`torch._int_mm` fica **pior que o bf16** porque obriga a quantizar fora, multiplicar, e reescalar
fora — tres kernels, dois trafegando bf16. Se algum lugar deste repo usa `_int_mm` como referencia
de "int8 nao acelera", a referencia esta' furada; o problema e' a API, nao o formato.

A nota completa: `github.com/JoaoZaokk/speech-quant-lab` -> `notes/acelerar-o-treino.md`.

---

