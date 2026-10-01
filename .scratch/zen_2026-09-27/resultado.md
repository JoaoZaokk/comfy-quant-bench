# Resultado: zen-image-edit como text encoder do Qwen-Image 2.1 (2026-09-27)

Critério em `criterio.md`. RTX 3090, ComfyUI de teste isolado (transformers 5.17.0 em `C:\ComfyBench\zen_test\pydeps`,
só o nó `zen-image-edit-comfyui` @ 7a29e58, adapter_v12, Qwen3.5-0.8B bf16, `HF_HUB_OFFLINE=1`). DiT e VAE originais.
Referência: bateria `bf16` (mesmo DiT, TE nativo Qwen3-VL-8B **em W4A8**).

| P | previsão | medido | veredito |
|---|---|---|---|
| Z1 | 12 imagens sem erro | 12/12, 1,04-1,08 it/s (= referência 1,05; o DiT é o mesmo) | confirmada |
| Z2 | MS-SSIM 0,4-0,7 vs ref | **0,841 média, 0,703 mín** — composição muito mais próxima do que previ. Para escala: DiT Q8_0 0,987, DiT Q4_1 0,912 | refutada para melhor |
| Z3 | >= 10/12 aderentes, sem artefato grosseiro | 11/12. "SLOW MORNINGS" 2/2 certo. Letreiro "QWEN IMAGE 2.1": seed 42 legível mas com uma linha extra de texto ilegível; seed 7 sai "9WEN" (Q vira 9). Referência: s42 limpo, s7 "21" sem ponto. Contagem de frutas: igual à referência (ambos erram em s7) | confirmada |
| Z4 | TE ~2,4 GB | encoder 1,7 GB + adaptador 0,64 GB (bf16), fora do gerenciador do ComfyUI; nativo W4A8 6,3 GB, BF16 17,5 GB | confirmada (tamanho em disco; VRAM não medida à parte) |

Folhas: `folha_s42.jpg`, `folha_s7.jpg` (em cima referência, embaixo zen). MS-SSIM por imagem em `msssim.json`.

## Não coberto

- Referência com TE nativo BF16 (não temos no disco); edição com imagens de referência; outros idiomas (declarado: só inglês).
- Arc: o encoder roda via transformers (PyTorch), deve funcionar em XPU, não testado.
- Para usar no ComfyUI do dia a dia: o ambiente principal precisa de transformers >= 5.17 (hoje 4.57.6) — decisão do dono.
