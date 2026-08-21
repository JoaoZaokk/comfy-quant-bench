# Re-medir o PV-NT sob janela limpa

Type: task
Status: open
Blocked by: 01

## Question

O patch PV-NT está provado **correto** e isso não precisa de nova medição: o braço NT
chega ao piso do f32 (1,530e-7 a 2,050e-7) contra referência f64 nas quatro formas que
o render emite, enquanto o braço NN existente fica em 2,623e-4 a 2,894e-4 em três das
quatro — 1406x a 1714x mais longe.

O que **não** está estabelecido é o ganho de tempo no nível do passo:

```
values  OFF: 15,33 / 13,18 / 11,90 / 12,13 / 12,71   -> 11,90-15,33 s
values  ON:   5,32 /  5,19 /  5,03                   ->  5,03- 5,32 s   (sem sobreposição)

passo   OFF: 66,5 / 60,0 / 56,1 / 57,3               -> 56,1-66,5 s
passo   ON:  56,2 / 53,0 / 56,5 / 54,4               -> 53,0-56,5 s   (SOBREPÕE)
```

A sub-fase separa; o passo não, porque a dispersão do braço OFF sozinho (1,19x)
engole a diferença. As quatro corridas foram alternadas, mas com a máquina em uso.

## Critério de fechamento

Fecha com um número de passo **e a dispersão dele**, sob janela acordada, ou com o
registro de que esta bancada não resolve um efeito desse tamanho — que também é
resposta, e uma resposta útil.

**Não** fecha com uma média sem dispersão. Uma corrida não é medição, e este projeto
já viu 1,4x entre corridas idênticas na mesma placa ociosa.
