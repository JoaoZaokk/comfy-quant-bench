#!/bin/bash
# Mata `colab exec` de PROBE (probe_qat.py / probe_ds.py) presos ha mais de 240 s. O --timeout do CLI nao os
# derruba (visto 4x em 24-25/09); o supervisor so' segue quando o exec morre. Nao toca em outro exec.
while true; do
  ps -eo pid,etimes,args | awk '/colab exec/ && /probe_(qat|ds)\.py/ && !/awk/ && $2 > 240 {print $1, $2}' | while read pid t; do
    kill "$pid" 2>/dev/null && echo "$(date +%T) matei probe preso pid=$pid (${t}s)"
  done
  sleep 30
done
