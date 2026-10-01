# Render pareado depois do upgrade pip: mesmos grafos da validação triton (ON), 3090, lock próprio.
param([string]$Casos = '')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:pip_upgrade_render'
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $a = @('.scratch\pip_2026-09-29\valida_pip.py', 'depois')
    if ($Casos) { $a += $Casos }
    & .\python_embeded\python.exe -s @a
    Write-Host "valida_pip rc=$LASTEXITCODE"
} finally {
    $ErrorActionPreference = 'Continue'
    Release-GpuLock
    Write-Host "FIM $(Get-Date -Format T)"
}
