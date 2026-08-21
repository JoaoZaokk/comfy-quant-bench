# A trava anti-cache é um chute, e está duplicada em duas linguagens

Type: task
Status: open

## Question

`tools/comfy_run_workflow.py:344` decide se o ComfyUI realmente renderizou usando
`if wall < 5.0 and files:` — constante mágica, sem nome, dentro do bloco de
impressão, sem efeito no código de saída.

Dez linhas acima, o mesmo `main()` já calcula
`execution_start → execution_success` a partir do que o servidor reporta, e joga o
valor fora num `print`. **O servidor afirma a duração; o código a estima.**

E `F:\cortiq-cmf\run_e2e_comfy.ps1:79` reimplementa a mesma heurística, com o mesmo
`5` cravado, em PowerShell. A mesma trava de segurança duplicada através de uma
fronteira de linguagem, sem nada que mantenha os dois números iguais.

## Por que isso é correção e não estilo

Esta checagem existe porque uma passada com prompt idêntico voltou em 0,6 s com as
49 saídas listadas e `status success` — o cache de resultado por nó do ComfyUI. Lido
de fora, isso é "ComfyUI faz em 0,6 s o que o cortiq faz em 510 s", 850×, e teria
sobrevivido a qualquer conferência superficial. O servidor confirma:
`Prompt executed in 0.01 seconds`.

## Independente do ticket 05

O modelo tipado do formato API (`05`) **não** dissolve este achado: aqui não se trata
da forma do prompt, e sim de usar o dado do servidor em vez de um limiar. Fica na
fronteira, tomável em paralelo.

## Critério de fechamento

Fecha quando as três forem verdade:

1. A duração do lado servidor é valor de primeira classe no código, não um `print`.
2. A detecção de cache usa esse valor, não `wall`.
3. Um cache-hit sai por **código de saída próprio**, e `run_e2e_comfy.ps1` apaga a
   cópia dele passando a olhar só o código de saída.

Não fecha com decisão escrita: é a única trava do arquivo contra reportar um cache
como render, e ela já quase falhou uma vez.
