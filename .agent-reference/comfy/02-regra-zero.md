# Regra zero e exemplos

> Referência preservada do CLAUDE.md original, linhas 86–135, coletado em 23/09/2026.
> Números, versões, estados e resultados são relatos datados; não foram reexecutados nesta preparação.
> Leia o trecho inteiro: correções posteriores podem limitar frases anteriores. Comandos históricos não autorizam execução.
> Caminhos em código e texto simples continuam relativos à raiz F:/COMFY_PORTABLE; links Markdown relativos foram rebasedos para esta pasta.

## REGRA ZERO: nunca assumir. Testar, ou pesquisar, antes de afirmar.

Posta pelo dono em 2026-09-01, depois de ele me corrigir **quatro vezes num dia**. Fica no topo
porque é a regra que todas as outras abaixo pressupõem, e porque eu a quebrei justamente enquanto
escrevia sobre rigor.

**Os dois modos de falha são diferentes, e o segundo é mais traiçoeiro:**

1. **Afirmar sem medir.** "Ninguém põe modulação em 4 bits, logo não se deve." Isso é consenso
   usado como evidência — três fontes concordando pode significar que as três copiaram a mesma
   suposição. Medido: `adaLN_modulation` é a camada com o **menor** erro em W4A4 de todo o bloco
   (0,1263 contra 0,1569 das demais). O argumento não existia.

2. **Declarar um limite e PARAR, em vez de procurar a volta.** Escrevi "2:4 não executa nesta
   máquina" e encerrei. O dono respondeu: *"executa, aceita A40, que é anterior mas mesma série da
   3090. O que falta é o kernel, e eu não vi você mandando ninguém procurar."* Ele estava certo: A40
   é **GA102, o mesmo silício** da 3090. Eu tinha um agente disponível e não usei. **Um limite que
   você não tentou contornar é uma hipótese, não um fato.**

   **Fechado por medição em 2026-09-01, e a hipótese era falsa em toda linha.** 2:4 executa nesta
   3090 a **1,7x–1,95x sobre o denso bf16**, medido em duas passadas independentes nos shapes reais
   do Z-Image (`bench/sparse24_na_sm86_2026-09-01.md`). Não faltava kernel: o kernel estava dentro
   do wheel Windows do xformers o tempo todo. Faltava um `tile` que coubesse — a config que eles
   compilam pede **139.264 bytes** de memória compartilhada e esta placa aceita **101.376**, porque
   foi dimensionada para a A100. Mesmo tile com dois estágios pede 69.632 e roda. E o `cuSPARSELt`,
   que três ferramentas deste repo culpavam por escrito, **não é usado pelo CUTLASS** e nunca foi o
   obstáculo. Três afirmações, todas erradas, todas na forma "não dá" — que é exatamente a forma que
   esta regra proíbe.

Mais dois do mesmo dia, para mostrar que não é caso isolado: (a) medi que Hadamard destrói o padrão
2:4 e chamei de achado — era **tautologia**, rotação densa preenche zero de qualquer matriz, e
faltava o controle; (b) disse "esparsidade custa 3x a quantização" comparando **erro de peso** de um
contra **erro de saída** do outro, e usando o método de poda que ninguém sério usa. Com critério
guiado por ativação a ordem **inverte**.

**Como aplicar, mecanicamente:**

- Antes de escrever "não dá", "não existe", "não suporta": rodar o teste, ou mandar um agente
  procurar. Custa minutos; a afirmação errada custa a confiança em tudo que veio junto.
- Antes de escrever "todo mundo faz assim, logo": perguntar se todo mundo **mediu**, ou se todo
  mundo **copiou**.
- Toda comparação carrega a métrica no nome. Erro de peso e erro de saída não se comparam, e foi
  assim que uma conclusão inteira nasceu errada.
- Todo conjunto de braços precisa do braço que pode falhar — o controle. Sem ele, um resultado bom
  não se distingue de sorte.

Ver as memórias [[prove-before-asserting]] e [[escrevo-mais-rapido-do-que-confiro]].

---

