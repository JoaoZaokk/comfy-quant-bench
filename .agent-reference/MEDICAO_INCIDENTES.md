> Preservado do CLAUDE.md global em 23/09/2026. Datas, medições e relatos abaixo são históricos; não representam revalidação atual. Regras específicas do projeto continuam aplicáveis.

## Como a REGRA ZERO falha na pratica (2026-09-21/22, quatro vezes num dia)

Nenhuma dessas foi "eu deduzi". Em todas eu **medi** — e a medicao mentiu.

1. **Conferir a UNIDADE antes da grandeza.** Reportei 88% de um dataset
   contando LINHAS de journal; eram 41.955 linhas para 36.068 itens unicos.
   Linha nao e' clipe, tarefa tentada nao e' clipe gerado, arquivo baixado
   nao e' arquivo aberto. -> memoria [[indicador-indireto-nao-e-medicao]]
2. **Amostra pequena superestima, e sempre para cima.** Projetei vazao de
   smokes de 30 clipes tres vezes e errei ~2x nas tres; cada erro custou
   re-fatiar a frota inteira. Smoke diz se FUNCIONA, nao quanto rende.
   E quando nao der para medir em producao, **fazer a divisao se
   auto-corrigir** em vez de tentar acertar a conta.
   -> memoria [[amostra-pequena-superestima-vazao]]
3. **Regua casada.** Comparei RTF de clipes de 13,5 s com clipes de 6,5 s e
   inflei uma das maquinas. Com custo fixo por chamada, RTF varia com a
   duracao. -> [[custo-por-segundo-nao-e-constante]], [[medir-com-a-regua-errada]]
4. **Ferramenta devolve sucesso sem ter feito o trabalho.** `sed` com ancora
   que nao casa sai rc=0 com zero substituicoes; excecao nao tratada mata o
   processo sem escrever no proprio log. **Depois de editar, conferir o
   arquivo. Depois de lancar, conferir o desfecho — nao a partida.**
5. **Instrumento certo, momento errado.** Li `nvidia-smi` em 0% durante o
   carregamento de 14,7 GB e declarei falha; o processo estava vivo. Uma
   leitura so' nunca fecha diagnostico — o segundo caminho fechou em 5 s.

**E a regra de engenharia que saiu disso:** quando o recurso pode sumir sem
aviso (VM, cota, sessao alheia), **o intervalo entre gravacoes E' a perda
maxima**. Subi um checkpoint de 45 para 90 min para economizar download e
perdi uma hora de A100 quando o Colab recolheu as placas. O conserto certo
era tirar a contencao do download, nao gravar menos.
-> memoria [[intervalo-de-gravacao-e-a-perda-maxima]]
