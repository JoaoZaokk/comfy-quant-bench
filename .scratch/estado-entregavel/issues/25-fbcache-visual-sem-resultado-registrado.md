# `fbcache_visual.py` é o único `fbcache_*` sem resultado registrado

Type: task
Status: resolved

## Question

`tools/fbcache_visual.py` existe (docstring: "End-to-end FBCache A/B on HunyuanVideo 1.5 with real
conditioning and decoded frames") e tem `argparse` + `main()`, então não é morto pela regra do
ticket ("chamado só pelo usuário na linha de comando não é morto"). Mas é o único dos 8 arquivos
`fbcache_*` que não aparece em `FBCACHE_FINDINGS.md`:

```
$ grep -n "fbcache_visual" FBCACHE_FINDINGS.md
(nada)
```

Comparar com a tabela de cobertura do mesmo documento (`FBCACHE_FINDINGS.md:302-312`), que lista
os outros 7:

| arquivo | listado em FBCACHE_FINDINGS.md |
|---|---|
| `fbcache_routing_test.py` | sim, linha 304 |
| `fbcache_catorder_test.py` | sim, linha 305 |
| `fbcache_tuple_test.py` | sim, linha 306 |
| `fbcache_branch_test.py` | sim, linha 307 |
| `fbcache_kwargs_test.py` | sim, linha 308 |
| `fbcache_clone_test.py` | sim, linha 309 |
| `fbcache_probe.py` | sim, linha 311 |
| `fbcache_audit.py` | sim, linha 311 |
| **`fbcache_visual.py`** | **não** |

`fbcache_visual.py` só é mencionado de um lado — o próprio `fbcache_probe.py` cita "The threshold
sweep in fbcache_probe.py..." dentro do docstring de `fbcache_visual.py`, mas nada aponta o
inverso. Não há evidência de que já rodou, nem resultado (quality/relL2/tempo) registrado em
lugar nenhum do repo.

Isto não é "achado de bug" — é achado de **cobertura**: uma ferramenta cara (carrega Qwen2.5-VL +
ByT5 + o transformer, decodifica pixels reais) que existe há tempo suficiente para os 7 irmãos
terem virado uma tabela de achados, e ela não.

## Critério de fechamento

Fecha com uma decisão escrita — ao contrário da maioria dos tickets deste lote, este não é
código quebrado, é lacuna de execução:

- ou alguém roda `fbcache_visual.py` (precisa de GPU e do checkpoint HunyuanVideo 1.5) e o
  resultado entra em `FBCACHE_FINDINGS.md` junto dos outros 7;
- ou fica documentado explicitamente por que não vale a pena rodar (redundante com
  `fbcache_probe.py` + `fbcache_audit.py`, por exemplo) e a decisão fica escrita no próprio
  docstring do arquivo.

Qualquer uma das duas fecha o ticket. Ficar como está — sem rodar e sem decisão — não fecha.

## Resolucao

Fechou pelo **primeiro** ramo do criterio: rodou, e o resultado entrou em `FBCACHE_FINDINGS.md`
junto dos outros sete (secao "HunyuanVideo 1.5", subsecao "O mesmo modelo, olhando o pixel em vez
do latente"). O `fbcache_visual.py` tambem entrou na linha de Ferramentas, que antes citava so o
`probe` e o `audit`.

**EXECUTADO** duas vezes, 3090 ociosa sob lock `bench:ticket25_fbcache_visual`, invocacao
identica nas duas (a segunda so com `--out` diferente):

```
.\python_embeded\python.exe -s .\tools\fbcache_visual.py
.\python_embeded\python.exe -s .\tools\fbcache_visual.py --out F:\COMFY_PORTABLE\_fbcache_visual_run2
```

| threshold | s (min-max) | speedup | hits | relL2 no pixel |
|---|---|---|---|---|
| baseline | 57,8 - 59,0 | 1,00x | - | - |
| 0,12 | 30,5 - 31,1 | 1,86 - 1,93x | 18/40 | 0,2363 |
| 0,20 | 20,0 - 20,2 | 2,86 - 2,95x | 26/40 | 0,2580 |

Duas corridas, spread de ~2% no tempo; `relL2` e `hits` identicos (o seed fixa). Reproduz por
caminho independente a medicao que o `fbcache_probe.py` ja tinha (1,89x / 3,06x) em outra
resolucao e outro arquivo.

**O achado que so esta ferramenta podia dar:** o `relL2` do latente nunca esteve otimista --
0,2157/0,2639 no latente contra 0,2363/0,2580 no pixel, praticamente o mesmo numero. O que
faltava era **olhar**. Em 0,20 a imagem e outra imagem: a maca perde a textura de casca inteira,
a mesa em diagonal com veio de madeira vira uma tabua chapada, o bokeh vira borrao uniforme. Em
0,12 a textura sobrevive mas a composicao ja mudou. "relL2 0,24" nao e degradacao leve; e uma
foto diferente pelo mesmo prompt.

**O que ficou sem cobertura:** um prompt, um modelo, uma resolucao, dois thresholds. Nada aqui
se estende a FLUX/Wan/LTX. E a leitura visual e minha, a olho, sobre as folhas de contato -- nao
ha metrica perceptual (LPIPS/SSIM) por tras dela; o que esta medido e tempo, hits e relL2.
