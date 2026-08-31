# RODADA 6: GitHub Issues e Plataformas Chinesas — 2026-08-30

## ÂNGULO A: GitHub Issues
**PR #14859 (Comfy-Org/ComfyUI):** suporte nativo ConvRot INT4 com fallback int8 automático por GPU.
**Issue #14721:** INT4 per-row quantization extends int8_tensorwise.
**Issue #582 (ComfyUI-nunchaku):** H100 (sm90) sem 4-bit tensorcores; fallback Turing/Ampere/Ada.
**Nenhuma issue:** "RTX 3090 / Ampere sm86 não suportado" — suporte documentado como presente.
**Controle:** "ComfyUI stable diffusion" retorna múltiplos resultados; OK.

## ÂNGULO B: Plataformas Chinesas (知乎, CSDN, juejin, oschina)
**Zhihu:** ConvRot W4A4 oficial; Nunchaku + SVDQuant reduz Flux para 12GB VRAM.
**CSDN:** NF4/INT8/FP8 quantização (40% VRAM savings); sem W4A4 fallback específico; Ampere não bloqueado.
**Juejin.cn:** Redirecionou para GitHub; sem cobertura nativa.
**Oschina.net:** Sem resultado direto; Zhihu dominou.

## Bloqueios
Nenhum: 403/captcha/login não ocorreram.

## Não encontrado
- Falha kernel ConvRot em sm86 (Ampere)
- Recusa W4A4 em RTX 3090 por falta de tensorcores
