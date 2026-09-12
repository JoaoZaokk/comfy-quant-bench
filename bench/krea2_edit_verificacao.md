# Krea2 Edit funciona: 4 de 4, e cada uma mudou so o que foi pedido

Executado 2026-09-12 na 3090. Nao e leitura de codigo: sao quatro renderizacoes pelo servidor
real, com o workflow oficial `krea2_identity_edit`, e as imagens estao em
`bench/krea2_edit_4prompts.png`.

## O desenho do teste

Quatro **categorias diferentes** de instrucao, nao quatro variacoes da mesma -- se as quatro
saissem iguais, ou as quatro ignorassem a instrucao, isso seria informacao sobre o teste, nao
sobre o modelo.

    braco      instrucao                                            semente
    roupa      "Change her outfit to a red raincoat."                1000
    noite      "Make it night time, with a full moon and stars."     1001
    insercao   "Add a small brown dog sitting on the grass."         1002
    fundo      "Replace the grassy hill with a sandy beach..."       1003

**O controle e a propria imagem de origem.** Edicao com preservacao de identidade nao tem braco
BF16 de referencia: o veredito e "mudou o que foi pedido e so isso", e isso so se le lado a lado.
Por isso a origem ocupa a primeira celula da folha de contato.

## Resultado

    braco      status    tempo     arquivo
    roupa      success   181,3 s   krea2_identity_edit_00002_.png
    noite      success   ~180 s    krea2_identity_edit_00003_.png
    insercao   success   124,9 s   krea2_identity_edit_00004_.png
    fundo      success   182,9 s   krea2_identity_edit_00005_.png

Olhando as cinco lado a lado -- e este julgamento e de quem olha, nao de nenhuma metrica:

- **roupa**: vestido rosa -> capa de chuva vermelha com capuz, cordao, botoes e bolsos. Rosto,
  cabelo, pose, bracos e fundo preservados. Os bracos passam a estar dentro das mangas.
- **noite**: ceu azul -> noite com lua detalhada e estrelas, grama escurecida, nuvens mantidas e
  repintadas como iluminadas pela lua. **O sujeito nao foi tocado** -- vestido ainda rosa, mesma
  pose.
- **insercao**: cachorro marrom acrescentado na grama, no **mesmo estilo de desenho** da origem
  (preenchimento chapado, mesmo peso de contorno). Sujeito intacto.
- **fundo**: grama -> praia com areia, mar, espuma e horizonte, com pegadas na areia. Sujeito
  intacto, mesma pose, mesmo lugar no quadro.

Nas quatro, o **estilo** da origem sobreviveu: continua o mesmo desenho infantil chapado, nao
virou foto nem ilustracao de outro tipo.

## O defeito que este teste encontrou, e que nao era do modelo

A primeira versao do script consultava **duas portas fixas** (o pai 8190 e o worker 27715). A
segunda edicao caiu no worker **27716** -- `ComfyUI-MultiGPU` sobe um processo por device e o
`/history` do resultado e o **daquele worker**. O script ficou pendurado ate o timeout enquanto a
imagem ja estava gravada em disco, e teria reportado `TIMEOUT`, que le como "o modelo nao gerou".

Duas das quatro edicoes (`noite` e `fundo`) sairam do 27716. Ou seja: **metade deste resultado
era invisivel para o cliente anterior.** Corrigido em `tools/comfy_run_workflow.py:bases_irmas`,
que descobre os servidores lendo a linha de comando dos processos em vez de assumir portas, e
dentro do laco de poll, porque o worker nasce sob demanda.

## Nao coberto

- **Uma imagem de origem, e ela e um desenho infantil chapado, nao uma foto.** Nada aqui diz o
  que o modelo faz com um rosto real, com pele, com textura fina ou com um fundo fotografico.
- **Uma semente por instrucao.** Nao ha replica.
- **Ninguem mediu identidade.** Nao ha metrica de rosto, de pose ou de estilo; quem diz que a
  pessoa continua a mesma e alguem olhando.
- Tudo isto rodou sobre o `krea2_turbo_int8_convrot` (o build int8 publico) com o LoRA
  `krea2_identity_edit_v1_2`. **Nada aqui diz o que a nossa quantizacao faz com a edicao** -- o
  criterio `bench/criterio_quant_krea2.md` deixa isso explicitamente de fora.
