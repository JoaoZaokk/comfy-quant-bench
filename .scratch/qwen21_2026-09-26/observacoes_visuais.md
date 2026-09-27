# Observações visuais minhas (não cegas; olhei sabendo o build) — bateria base, 26/09 ~19:15

Folhas em `folhas_fase1/` (512 px por build; `_corte` = centro 30% ampliado).
- p0_s42 (letreiro neon): bf16, int8 e mixed escrevem "QWEN IMAGE 2.1" limpo. **int4: traço duplo no neon ("Q"
  com contorno dobrado), espaçamento quebrado "IMAGE2.1"** — o artefato neon já visto antes.
- p0_s7 corte (vidro/interior): int8 ≈ bf16; mixed muda cor (tom lilás) e suaviza; int4 muda estrutura (moldura/
  costura extra no vidro).
- p1_s42 corte (pele): int8 ≈ bf16 (poros e barba iguais); mixed um pouco mais liso e outra expressão; **int4
  perde textura de pele (mais liso, "plástico")**.
- p2_s7 (pôster): bf16/int8/mixed põem título pequeno no topo; int4 compõe centralizado e maior (outra trajetória;
  não é defeito, sem texto errado).
- p5_s42 (3 maçãs, 2 peras, mão): bf16 certo (3 maçãs). **int8: a mão segura meia maçã/fatia estranha (2 maçãs
  inteiras + objeto)**; mixed certo (3); **int4: só 2 maçãs**. Uma amostra — trajetória, não prova de defeito do
  build, mas mostra que PSNR alto (int8) não protege contagem/objeto.
