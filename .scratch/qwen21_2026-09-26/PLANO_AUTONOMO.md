# Plano autônomo Qwen-Image-2.1 (pedido do dono, 26/09/2026 ~18:40, antes do cochilo)

Pedido literal: "faça todos os testes, braços, testes, treino e geração, use o mimo no opencode para avaliar as imagens,
mimo 2.6 pro, e continue, quero tudo testado, positivos, negativos, quantizações, o que deu bom, deu ruim, o que fizemos
diferente, o que podemos fazer, o que você testou, por exemplo, comparar a sua quant com a do comfy, etc, testes que rodou
comparando por camadas, melhorias de runtime, onde você encontrou erros no processo do comfy, onde melhorou, onde não
melhorou e por que (de um motivo razoável, updates controlados e etc estão aprovados)."

## Estado ao gravar
- ComfyUI 0.37.4 atualizado e validado (Eros idêntico). Commit fc1b343 pushado. Repos: comfy-stream-video-save,
  comfy-qwen21-nunchaku (JoaoZaokk).
- BF16 Comfy-Org baixado (P:/ComfyBench/diffusion_models/qwen_image_2.1_bf16.safetensors, confere).
  Junção dos shards: qwen_image_2.1_bf16_junto_dos_shards.safetensors. Comparação: 233/233 tensores comuns idênticos;
  Comfy-Org fundiu gate_layer+proj em gate_up (cat nessa ordem, 32/32 blocos, byte a byte). Registrar.
- Bateria de 60 imagens rodando (`roda_bateria.ps1`, grafos em `bateria/`, saída ComfyUI/output/qwen21_bateria/<dit>/),
  DiTs bf16, bf16junto, int8, mixed, int4. Critério em `criterio.md`. Métricas: tools/metricas_imagem.py (sem LPIPS:
  pesos VGG não em cache; baixar pede autorização... o dono aprovou "updates controlados" — LPIPS é download de pesos,
  decidir com parcimônia).

## Fila (em ordem)
1. Métricas da bateria vs bf16 (PSNR/SSIM/MS-SSIM/grão), it/s e VRAM por DiT; bf16 x bf16junto (fused vs não).
2. Avaliação das imagens com MiMo 2.6 Pro via opencode (achar config/CLI do opencode; imagens SFW do Qwen). Rubrica fixa,
   cega ao nome do build se possível.
3. NOSSAS quantizações do DiT com as ferramentas da bancada (tools/quant_w4a8.py, quant_w4a4.py, quant_int8.py,
   quant_mixed.py): conferir suporte à arquitetura qwen_image21 (loader/config antes de estender — regra). Gerar
   W4A8, INT8 ConvRot, W4A4 ConvRot a partir do BF16 (fonte: junto ou Comfy-Org; saída ao lado, sufixos, sidecar).
   Comparar com int8 Comfy-Org e mixed NidAll: por camada (erro rel. vs BF16, tools existentes de análise por camada),
   render/métricas, it/s, VRAM.
4. Runtime: flags/opções (dynamic VRAM x --disable-dynamic-vram nos nossos builds em 0.37.4 — o .bat do dono não usa a
   flag), shift fixo 0,69 do ComfyUI (issue #16447, PR #16553) a 2048², prefix KV cache/compile (master tem 2f7c6d47,
   1d61dcc3; a 0.37.4 não). Updates controlados aprovados pelo dono.
5. Treino: ATENÇÃO — AGENTS/memória registram veto do dono a treino QAT na RTX 3090 (23/09, klein). O pedido atual diz
   "treino"; interpretar como QAT só-escalas (receita que ganhou no klein) preferencialmente no Colab se houver quota;
   na 3090 só se ficar claro que o dono liberou — registrar a decisão e o motivo no relatório.
6. Relatório final para o dono (artifact/HTML ou md): positivos, negativos, o que fizemos diferente do Comfy/NidAll/
   mesmertech, erros do processo do ComfyUI encontrados (CUDA_VISIBLE_DEVICES, shift fixo, validação que "passa" sem
   rodar, estimativa de memória LTXAV, fp16-intermediates que muda áudio, decode que materializa o vídeo), onde melhorou,
   onde não melhorou e por quê. Atualizar W4A4_PROGRESS, commitar e pushar (dono autorizou commit/push nesta rodada).

## Regras que continuam valendo
- NSFW (Eros): só números. Qwen: imagens SFW, pode olhar.
- Lock GPU (Assert/Release), CUDA_VISIBLE_DEVICES=0,1, COMFYUI_MGPU_DISABLED=1 nos drivers, --disable-dynamic-vram
  nos testes (registrar quando testar sem).
- Nunca sobrescrever/mover originais; saídas novas ao lado, .partial -> os.replace.
- F: tem ~14 GB livres; modelos novos vão para P: ou D:.
