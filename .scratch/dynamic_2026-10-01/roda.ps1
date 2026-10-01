param([string]$Bracos = 'dyn,nodyn', [string]$Casos = '')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:nunchaku_dynamic_vram'
try {
    foreach ($b in $Bracos.Split(',')) {
        $a = @('.scratch\dynamic_2026-10-01\teste_dynamic.py', $b)
        if ($Casos) { $a += $Casos }
        & .\python_embeded\python.exe -s @a
        Write-Host "$b rc=$LASTEXITCODE"
    }
} finally {
    $ErrorActionPreference = 'Continue'
    Release-GpuLock
    Write-Host "FIM $(Get-Date -Format T)"
}
