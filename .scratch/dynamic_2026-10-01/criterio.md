# Nunchaku com dynamic VRAM (2026-10-01): critério, escrito antes de medir

Pergunta: os loaders Nunchaku (Z-Image e Qwen-Image) ainda quebram com dynamic VRAM ligado no ComfyUI 0.37.4?
A nota de 2026-08-18 (09-comandos-launchers.md) diz que sim: `AttributeError: 'NoneType' object has no attribute 'dtype'`.

Teste: ComfyUI próprio na 8199, só a 3090, MGPU desligado, flags do launcher (sage + triton).
Braços: `dyn` (sem --disable-dynamic-vram) e `nodyn` (com). Casos:
- NZ: NunchakuZImageDiTLoader svdq-int4_r32-z-image-turbo, 1024², 8 passos, seed 5.
- NQ: NunchakuQwenImageDiTLoader nunchaku_qwen_image_edit_2511_best_quality_int4 como t2i, 1024², 8 passos, seed 42.
Cada caso frio e quente.

Aprovado para dynamic se: success nos dois casos E imagem igual ao braço nodyn (idêntica ou PSNR > 40 dB).
Se falhar: registrar o erro e a camada, achar a causa no código, corrigir e repetir o mesmo teste.

## Adendo (2026-10-01 12:55), escrito antes dos controles
Rodada 3: 8/8 success, mas dyn x nodyn não bateu (NZ PSNR 13-16 dB, NQ 30-36 dB). Visualmente as duas boas.
Controles: (a) repetir os dois braços (`_rep`) para medir determinismo dentro do braço; (b) Z1 = Z-Image comum
(UNETLoader bf16) nos dois braços. Se Z1 também diverge dyn x nodyn e cada braço repete igual a si mesmo, a diferença
é do caminho dynamic do ComfyUI (TE/VAE/DiT comum), não do conserto do Nunchaku.
