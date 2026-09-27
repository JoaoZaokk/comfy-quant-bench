#!/bin/bash
# Sobe a VM de GPU (G4; se falhar, A100), prepara, baixa o BF16, lanca o QAT em background. Um dono do CLI:
# depois disto so' o supervisor fala com a sessao.
C=/home/pipeline/.local/bin/colab; W=/mnt/f/COMFY_PORTABLE; D=$W/.scratch/qat_qwen21_2026-09-27; T=$W/tools/colab_qat_qwen21
S=${1:-qwq-g4}; GPU=${2:-G4}
export REQUEST_TIMEOUT=900
timeout 900 $C new -s $S --gpu $GPU 2>&1 | grep -v '^\[colab\]' | tail -3
timeout 180 $C exec -s $S --timeout 120 -f $D/celula_pastas.py 2>&1 | grep -v '^\[colab\]' | tail -1 || exit 1
for par in $D/comfy_v0374.zip:comfy_v0374.zip $D/config_gpu.json:config.json $D/conds_qwen21.pt:conds_qwen21.pt \
           $T/qat_qwen21_blocos.py:qat_qwen21_blocos.py $T/celula_prepara.py:celula_prepara.py $T/celula_lanca.py:celula_lanca.py; do
  timeout 600 $C upload -s $S ${par%%:*} /content/qatq/${par##*:} 2>&1 | grep -v '^\[colab\]' | tail -1; done
# token de escrita do HF (conteudo nao e' lido aqui)
timeout 120 $C upload -s $S $W/.hf/token /root/.cache/huggingface/token 2>&1 | grep -v '^\[colab\]' | tail -1
echo "== prepara"; timeout 2400 $C exec -s $S --timeout 2300 -f $T/celula_prepara.py 2>&1 | grep -v '^\[colab\]' | tail -25
echo "== lanca"; timeout 300 $C exec -s $S --timeout 240 -f $T/celula_lanca.py 2>&1 | grep -v '^\[colab\]' | tail -3
