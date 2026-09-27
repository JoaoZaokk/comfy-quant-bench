# Lock da GPU do inicio ao fim: teste Triton (P2), um boot do ComfyUI e a bateria da fase pedida.
param([string]$Fase = 'principal', [string[]]$Extra = @(), [string]$Porta = '8191', [string]$Cvd = '0,1')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\lowbit_2026-09-27'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner "comfy:lowbit_loader_$Fase"
try {
    $env:CUDA_VISIBLE_DEVICES = $Cvd
    if ($Fase -eq 'principal') {
        & .\python_embeded\python.exe -s custom_nodes\comfy-lowbit-loader\test_lowbit.py
        Write-Host "=== testes rc=$LASTEXITCODE"
    }
    $env:COMFYUI_MGPU_DISABLED = '1'
    $a = @('-s', '.\ComfyUI\main.py', '--windows-standalone-build', '--use-sage-attention', '--listen', '127.0.0.1', '--port', $Porta) + $Extra
    $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
         -RedirectStandardOutput "$D\comfy_$Fase.out.log" -RedirectStandardError "$D\comfy_$Fase.log"
    try {
        & .\python_embeded\python.exe -s "$D\roda_lowbit.py" $Porta $Fase
        if ($LASTEXITCODE -ne 0) { Write-Host "ERRO rc=$LASTEXITCODE" }
    } finally {
        $ErrorActionPreference = 'Continue'
        & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            ? { [string]$_.CommandLine -match "main\.py.*--port $Porta" } |
            % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM $Fase $(Get-Date -Format T)"
