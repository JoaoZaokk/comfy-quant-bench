# Uma fase da validação sob lock: testes unitários de GPU (só na fase off) + matriz de grafos. 3090 apenas.
param([ValidateSet('off', 'on')][string]$Flag, [string]$Casos = '', [switch]$SemUnit)
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$D = '.scratch\triton_2026-09-29'
Assert-GpuLock -Owner "comfy:validacao_triton_$Flag"
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    if ($Flag -eq 'off' -and -not $SemUnit) {
        $env:KITCHEN_TEST_CUDA = '1'; $env:KITCHEN_TEST_DEVICE = 'cuda:0'
        $env:LOWBIT_TEST_CUDA = '1'; $env:LOWBIT_TEST_DEVICE = 'cuda:0'
        & .\python_embeded\python.exe -s -m pytest patches\tests\test_comfy_kitchen_awq.py -q *>&1 | Tee-Object "$D\unit_kitchen.log"
        & .\python_embeded\python.exe -s custom_nodes\comfy-lowbit-loader\test_lowbit.py *>&1 | Tee-Object "$D\unit_lowbit.log"
    }
    $a = @("$D\valida_triton.py", $Flag)
    if ($Casos) { $a += $Casos }
    & .\python_embeded\python.exe -s @a
    Write-Host "valida_triton rc=$LASTEXITCODE"
} finally {
    $ErrorActionPreference = 'Continue'
    Release-GpuLock
    Write-Host "FIM $Flag $(Get-Date -Format T)"
}
