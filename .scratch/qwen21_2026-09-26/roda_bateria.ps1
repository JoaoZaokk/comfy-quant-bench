# Um boot do ComfyUI e os grafos de bateria/ordem.txt em sequencia (agrupados por DiT). Lock da GPU do inicio ao fim.
param([string]$Ordem = 'bateria\ordem.txt', [string]$Dono = 'comfy:qwen21_bateria_metricas', [string]$Log = 'bateria_comfy', [switch]$Dinamico, [string[]]$Extra = @())
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner $Dono
try {
    $env:COMFYUI_MGPU_DISABLED = '1'
    $env:CUDA_VISIBLE_DEVICES = '0,1'
    $a = @('-s', '.\ComfyUI\main.py', '--windows-standalone-build', '--use-sage-attention', '--disable-dynamic-vram',
           '--listen', '127.0.0.1', '--port', '8190')
    $a += $Extra
    if ($Dinamico) { $a = $a | ? { $_ -ne '--disable-dynamic-vram' } }  # como o .bat do dono
    $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
         -RedirectStandardOutput "$D\$Log.out.log" -RedirectStandardError "$D\$Log.log"
    try {
        foreach ($g in Get-Content "$D\$Ordem") {
            if (-not $g) { continue }
            Write-Host "=== $g $(Get-Date -Format T)"
            & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\$g" "$D\bateria\amostra_$([IO.Path]::GetFileNameWithoutExtension($g)).csv"
            if ($LASTEXITCODE -ne 0) { Write-Host "ERRO rc=$LASTEXITCODE em $g" }
        }
    } finally {
        & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            ? { [string]$_.CommandLine -match 'main\.py.*--port 8190' } |
            % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM bateria $(Get-Date -Format T)"
