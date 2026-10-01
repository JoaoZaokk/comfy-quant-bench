#!/bin/bash
# Cadeia local da noite: replay final, depois b1..b4 na ordem em que a fila da VM os termina.
# Cada etapa espera o proprio artefato no HF, entao a cadeia pode ser lancada antes de tudo.
cd /f/COMFY_PORTABLE
bash .scratch/avalia_replay_final.sh 2>&1 | tee .scratch/avalia_replay_final.log
for b in b1 b2 b3 b4; do
  bash .scratch/avalia_braco.sh $b 2>&1 | tee .scratch/avalia_$b.log
done
echo "=== FIM avalia_noite $(date +%T) ==="
