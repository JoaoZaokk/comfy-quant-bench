# O braço BF16 do Qwen-Image-Edit é inalcançável nesta máquina pelo caminho do servidor

Escrito 2026-09-13, depois de quatro tentativas e de a máquina ficar parcialmente travada.
Registrado como **limite medido**, não como escopo cortado.

---

## O que se queria

Medir o caminho de **edição** do Qwen-Image-Edit 2511 — o que dá nome ao modelo — com três
braços: BF16 original, o `int8_convrot` do Comfy-Org, e o nosso `w4a8`. Isso existe porque a
medição publicada antes usou `quality_ladder.py`, que amostra do ruído: **text-to-image, sem
imagem de entrada**. O dono apontou o erro.

## As quatro falhas

Todas no mesmo grafo, a mesma imagem, a mesma instrução, variando só como o transformer é
carregado:

```
1. UNETLoader, placa conferida limpa em 287 MiB
   -> torch.AcceleratorError: CUDA error: out of memory
      levantado de torch.cuda.mem_get_info, dentro de model_management.free_memory

2. UNETLoaderDisTorch2MultiGPU, expert_mode_allocations "cpu,40gb"
   -> Windows fatal exception: access violation
      servidor morto; depois disso nvidia-smi e listagem de diretorio passaram de 120 s

3. --lowvram (o equivalente no servidor ao que o ladder faz sozinho)
   -> servidor travou no proprio arranque: 895 bytes de log em uma hora

4. depois do crash, `import torch` passou a travar por mais de 120 s
   -> nenhum servidor sobe; consultas WMI/CIM tambem travam
```

## O mecanismo, e por que ele transfere

No grafo de **edição** o transformer de 38,05 GiB tem de coexistir com o encoder
`qwen_2.5_vl_7b` de 15,45 GiB e ainda há um `VAEEncode` da imagem de entrada. No grafo de
text-to-image não há encoder de imagem nem `VAEEncode`, e o encoder de texto é liberado antes da
amostragem.

**O mesmo arquivo roda a 113,9 s por renderização no ladder de t2i.** Ele consegue porque
`quality_ladder.py` escreve `comfy_mm.vram_state = NORMAL_VRAM` **antes** de carregar
(`sample_all`, a regra `gib < 0.7 * total`). O servidor do ComfyUI não faz isso: ele decide a
política de colocação por conta própria e, neste grafo, decide errado.

> **Um checkpoint que sabidamente funciona pode ser inalcançável por outra porta de entrada, e a
> diferença não está no arquivo.** Esta bancada já tinha o caso inverso registrado — o
> `--reserve-vram` que parecia não vincular quando recomputado à mão e vinculava de verdade na
> chamada real. É a mesma lição: modelar o caminho não substitui percorrê-lo.

## Consequência para o que for publicado

A bateria de edição roda com **dois braços**, `int8_convrot` do Comfy-Org e o nosso `w4a8`.
Portanto:

- **A referência NÃO é o original.** É o int8, que é o braço que venceu em fidelidade em cinco
  famílias seguidas nesta bancada (Z-Image, MiniMax, epsilon por passo, Krea2, Qwen t2i) e
  também no LTX, onde era da própria Lightricks.
- Qualquer número de edição publicado diz **"distância até o int8 do Comfy-Org"**, nunca
  "distância até o original". Os dois braços podem estar igualmente longe do BF16 e isso não
  aparece.
- A comparação de **velocidade** entre os dois continua válida: ambos residem na placa, nenhum
  é espalhado.

## O que NÃO foi tentado

- Resolução menor que 1 megapixel na entrada. Reduziria a memória do `VAEEncode` e do latente,
  e mudaria o regime medido — seria outra medição, não a mesma com menos risco.
- Rodar a edição **em processo**, imitando o que o ladder faz (forçar `NORMAL_VRAM` antes de
  carregar). É o caminho mais promissor e não foi feito porque exigiria reimplementar oito nós
  que já existem e já foram exercitados aqui, e porque a máquina travou antes.
- `--novram`, que é mais agressivo que `--lowvram`.
- Uma segunda máquina.
