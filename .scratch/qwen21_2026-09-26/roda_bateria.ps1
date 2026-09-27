# Um boot do ComfyUI e os grafos de bateria/ordem.txt em sequencia (agrupados por DiT). Lock da GPU do inicio ao fim.
param([string]$Ordem = 'bateria\ordem.txt', [string]$Dono = 'comfy:qwen21_bateria_metricas', [string]$Log = 'bateria_comfy', [switch]$Dinamico, [string[]]$Extra = @(), [string]$Porta = '8190', [string]$Cvd = '0,1')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner $Dono
try {
    $env:COMFYUI_MGPU_DISABLED = '1'
        $env:COMFY_PORT = $Porta
    $env:CUDA_VISIBLE_DEVICES = $Cvd
    $a = @('-s', '.\ComfyUI\main.py', '--windows-standalone-build', '--use-sage-attention', '--disable-dynamic-vram',
           '--listen', '127.0.0.1', '--port', $Porta)
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
        # No Windows PowerShell 5 o stderr do taskkill ("not found") vira erro terminante com 'Stop' e pulava a
        # segunda limpeza, deixando o servidor vivo (27/09).
        $ErrorActionPreference = 'Continue'
        & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            ? { [string]$_.CommandLine -match "main\.py.*--port $Porta" } |
            % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM bateria $(Get-Date -Format T)"
