# Condicionamentos do QAT Qwen 2.1 na 3080 Ti (a 3090 esta com 17,8 GB presos por um processo zumbi do teste do aimdo).
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qat_qwen21_conds'
try {
    $env:CUDA_VISIBLE_DEVICES = '1'
    & .\python_embeded\python.exe -s tools\colab_qat_qwen21\gera_conds_qwen21.py --treino tools\colab_qat_qwen21\prompts_treino.txt `
        --holdout tools\colab_qat_qwen21\prompts_holdout.txt --out .scratch\qat_qwen21_2026-09-27\conds_qwen21.pt
    Write-Host "rc=$LASTEXITCODE"
} finally { Release-GpuLock }
