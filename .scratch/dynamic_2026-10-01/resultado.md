# Nunchaku com dynamic VRAM (2026-10-01): resultado

Critério em `criterio.md` (com adendo escrito antes dos controles). 3090, ComfyUI 0.37.4, porta 8199, sage + triton.

## Por que quebrava (lido no código e reproduzido)
1. **Lazy Linear.** Com dynamic VRAM (`aimdo_enabled`), `comfy/ops.py:523-544` monta `Linear` com `weight=None` e
   `bias=None` até o state dict carregar. Os loaders Nunchaku constroem o modelo com essas camadas e trocam por
   `SVDQW4A4Linear.from_linear()` antes de carregar, lendo `weight.dtype`, `weight.device` e `bias is not None`
   (nunchaku/models/linear.py). Resultado: `'NoneType' object has no attribute 'dtype'`, e o bias sumiria em silêncio.
   Conserto: `eager_linear_init()` em `ComfyUI-nunchaku/nodes/utils.py` desliga o lazy só durante a construção e o
   load nos loaders Z-Image e Qwen-Image. O modelo continua num ModelPatcher comum (não dinâmico), como antes.
2. **`fast_disk` no clone (independe do dynamic).** O ComfyUI 0.37.4 passa `fast_disk=` em `ModelPatcher.clone()`
   (`comfy/model_patcher.py:446`); `ZImageModelPatcher.__init__` não aceitava. Medido: TypeError no
   `ModelSamplingAuraFlow` (rodada 2, braço dyn). Pela leitura, quebraria igual sem dynamic (não medido no nodyn
   porque o conserto entrou antes). Conserto: `**kwargs` no `__init__` e no `clone`.
O preflight (`comfy-quant-preflight`) recusava todo Nunchaku com dynamic; agora deixa passar os dois loaders
consertados quando o patch está carregado e só avisa para os outros nós Nunchaku. 41/41 testes.
Patch: `patches/nunchaku_eager_linear_dynamic_vram.patch`.

## Medido
- Rodada 3 e controles: 20/20 success (NZ, NQ, Z1; dyn e nodyn; frio e quente; repetidos).
- Dyn x nodyn: NZ PSNR 13,3-16,0 dB; NQ 30,1-36,3 dB.
- **Mesmo braço repetido**: NZ 14,3-14,9 dB; NQ 31,3-33,2 dB. O Nunchaku não é determinístico entre execuções, e
  a diferença dyn x nodyn está dentro dessa dispersão. O critério "idêntica ou > 40 dB" estava errado para Nunchaku.
- Z1 (Z-Image comum): dyn x nodyn **idênticas pixel a pixel**, frio e quente.
- Visual (folha.jpg): as duas saídas bem formadas, sem artefato.
- Tempo quente igual: NZ 3,3-3,4 s, NQ 11,4-11,5 s, Z1 7,1-7,2 s. Frio variou muito (NZ dyn 15,5 e 123,9 s; NQ dyn
  26,3 e 67,1 s; nodyn 12,6-21,9 e 32,7-34,5 s); duas amostras, sem conclusão sobre frio.

## Não medido
- `NunchakuQwenImageLoraStackV3` (8 workflows salvos usam), Flux, text encoder, IP-Adapter e PuLID Nunchaku.
- LTX 2.5 com dynamic: continua precisando de `--disable-dynamic-vram` (OOM de 2026-09-30 no log do dono).
