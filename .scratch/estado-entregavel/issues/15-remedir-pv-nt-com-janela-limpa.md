# Re-medir o PV-NT sob janela limpa

Type: task
Status: resolved
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

## Resolução — 2026-08-21, janela limpa, lock `cortiq:ab_pv_nt`, 4 corridas alternadas OFF/ON/OFF/ON

`pwsh -NoProfile -File .\run_ab_pvnt.ps1` (sob `powershell` 5.1 o script morre no próprio log —
ver ticket `29`). Passos 2 e 3 apenas; o passo 1 compila pipelines e não é o regime.

**A sub-fase separa. O passo não. Igual antes — mas agora medido sob janela limpa, e os dois
números encolheram.**

| | OFF | ON | |
|---|---|---|---|
| `values n=1792 k=1792 m=128` | **5,31–5,83 s** | **3,11–3,54 s** | sem sobreposição, ~1,7x |
| passo (2 e 3) | 55,9–59,9 s | 56,9–66,9 s | **sobrepõe** |
| denoise total | 193,6–200,0 s | 185,2–195,4 s | sobrepõe |
| wall | 483,2 / 490,1 s | 486,2 / 496,7 s | sobrepõe |

**Fecha pelo segundo ramo do critério: esta bancada não resolve um efeito desse tamanho.** O ganho
da sub-fase é ~2,3 s por passo; o passo custa ~58 s. São **4%**, contra uma dispersão própria de
10 s no braço ON. Não há corrida suficiente aqui para separar isso, e não é falta de repetição —
é a razão sinal/ruído.

**O que a janela limpa mudou, e isso é o achado que viaja:** as medições de 2026-08-19, com a
máquina em uso, deram `values` OFF **11,90–15,33 s** e ON **5,03–5,32 s** — razão ~2,4x. Hoje,
placa ociosa: OFF **5,31–5,83** e ON **3,11–3,54** — razão **~1,7x**. Os dois braços ficaram
~2,3x mais rápidos, e **a razão entre eles caiu 30%**. A contenção não só inflou os tempos
absolutos, inflou o ganho aparente. Qualquer número de A/B tirado daquela sessão está para cima.

`GFLOP/s` do mesmo GEMM: OFF 216–238, ON 357–411.

**O caso para ligar `CMF_LTX_PV_NT=1` continua sendo precisão, não velocidade** — 1406x a 1714x
mais perto da referência f64, já provado e não re-medido aqui. Segue `off by default`.

**Não coberto:** latentes das quatro corridas foram gravados (`latent_{off,on}_r{1,2}.safetensors`)
e **não** comparados nesta resolução — a paridade já está estabelecida por `pv_nt_parity.rs` contra
referência f64, que é prova mais forte que comparar duas saídas do próprio motor. O passo 1
(compilação de pipeline) foi descartado por desenho e não foi caracterizado.
