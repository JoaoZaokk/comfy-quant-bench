#!/bin/bash
# Tenta A100 (hm, depois standard) a cada 15 min, ate 5 tentativas. Ao subir: ambiente, mkdir, uploads.
export MSYS_NO_PATHCONV=1
C="wsl.exe -d Ubuntu -u pipeline -- /home/pipeline/.local/bin/colab"
W=/mnt/f/COMFY_PORTABLE
L=/f/COMFY_PORTABLE  # a bancada vista por este Git Bash
. "${L:-$W}/.scratch/sobe_qat_comum.sh"  # sobe_qat: codigo do QAT de colab_ops.ARQUIVOS_QAT
for t in 1 2 3 4 5; do
  for shape in "--high-mem" ""; do
    echo "tentativa $t A100 $shape $(date +%T)"
    out=$($C new -s qat-a100 --gpu A100 $shape 2>&1 | tail -3)
    echo "$out"
    if echo "$out" | grep -q "Session READY"; then
      echo "SUBIU A100 $shape $(date +%T)"
      $C exec -s qat-a100 --timeout 180 -f $W/tools/colab_qat/celula_ambiente.py 2>&1 | grep AMBIENTE
      sobe_qat qat-a100 || exit 1
      for f in bench/qat_klein/prompts_treino.txt bench/qat_klein/prompts_holdout.txt .scratch/wheels_colab/diffusers-0.38.0-py3-none-any.whl tools/colab_qat/probe_qat.py; do
        $C upload -s qat-a100 $W/$f /content/qat/$(basename $f) 2>&1 | tail -1
      done
      echo "PRONTO_PARA_TOKEN $(date +%T)"
      # kernel ativo enquanto o dono poe o token: exec real a cada 4 min, ate 2 h
      for k in $(seq 1 30); do sleep 240; $C exec -s qat-a100 --timeout 60 -f $W/tools/colab_qat/celula_vivo.py 2>&1 | tail -1 | sed "s/^/vivo $k /"; [ -f /f/COMFY_PORTABLE/.scratch/a100_para_vigia ] && { echo "vigia parado"; exit 0; }; done
      exit 0
    fi
  done
  [ $t -lt 5 ] && sleep 900
done
echo "A100 NAO SUBIU em 5 tentativas $(date +%T)"
