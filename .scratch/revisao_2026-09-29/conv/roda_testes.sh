#!/bin/bash
# roda os testes da area de conversao, CPU so. uso: roda_testes.sh <rotulo>
cd /f/COMFY_PORTABLE
out=.scratch/revisao_2026-09-29/conv/testes_$1
mkdir -p $out
for t in test_conversion_core test_native_probe test_quant_mixed_provenance test_quant_mixed_sigma test_smooth_guards test_svdq_verify test_svdq_write_contract test_verify_formats test_avaliar_backend $EXTRA; do
  CUDA_VISIBLE_DEVICES=-1 timeout 900 python_embeded/python.exe -s tools/$t.py > $out/$t.log 2>&1
  echo "$t rc=$?"
done | tee $out/resumo.txt
