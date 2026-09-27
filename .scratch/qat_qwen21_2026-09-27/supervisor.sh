#!/bin/bash
# Supervisor com probe real de kernel a cada 4 min (unico dono do CLI enquanto roda).
S=${1:-qwq-g4}
export PATH="$HOME/.local/bin:$PATH"
CLI_PY=/home/pipeline/.local/share/uv/tools/google-colab-cli/bin/python
cd /mnt/c/Users/joaoz/projetos/project_quant_merge_frankestein/.worktrees/colab-gpu-workspace
"$CLI_PY" -u scripts/colab/colab_job_supervisor.py --session $S \
  --probe /mnt/f/COMFY_PORTABLE/tools/colab_qat_qwen21/probe_qat_qwen21.py --interval 240 --timeout 120
