# Atualização de pacotes pip do python_embeded (2026-09-29) — critério, escrito antes de medir

Pedido do dono: "atualizar, ver o que quebrou, ver o que conserta" (em vez de ficar preso a versão antiga).

## Fora do escopo (regra do projeto: sem upgrade em massa de Torch/CUDA/ComfyUI/comfy-kitchen)
torch, torchvision, triton-windows, comfy-kitchen (patch local), comfyui_frontend_package,
comfyui-workflow-templates*, comfyui-embedded-docs (fixados pelo requirements.txt do ComfyUI 0.37.4).
Qualquer pacote cujo upgrade puxe um desses é recusado (pip --dry-run antes).

## Segurança
- Cópia integral `python_embeded` → `python_embeded_backup_20260929` antes de mexer (rollback = trocar pastas;
  os .exe de Scripts do backup apontam para python_embeded, certo após a troca).
- `freeze_antes.txt` salvo.

## Aceitação
1. Importação: `main.py --quick-test-for-ci` (3090 com lock, MGPU desligado), lista de IMPORT FAILED e de
   tempos por nó. Depois do upgrade: nenhum nó novo falhando além dos que já falham hoje, ou falha corrigida.
2. Tokenizadores (CPU): ids de token dos tokenizadores do core (CLIP-L, T5, Qwen2.5-VL, Qwen3, Gemma, Llama, UMT5,
   Mistral/Byt5 se houver) para um conjunto fixo de prompts: iguais byte a byte antes x depois.
3. Render: workflows do dia a dia com semente fixa, antes x depois, mesma placa e flags. Esperado idêntico
   (transformers só entra via tokenizador no core); diferença exige explicação.
4. `pip check` sem conflito novo.
Estrutura/importação não aprova qualidade: o item 3 é o que aceita.
