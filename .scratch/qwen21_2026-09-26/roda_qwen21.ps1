# Um boot do ComfyUI 0.37.4 e os prompts do Qwen-Image-2.1 em sequencia. Lock da GPU do inicio ao fim.
param([string[]]$Prompts = @('int8_s42', 'mixed_s42', 'int4_s42'))
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:qwen21_primeiro_render'
try {
    $env:COMFYUI_MGPU_DISABLED = '1'
    $env:CUDA_VISIBLE_DEVICES = '0,1'
    $a = @('-s', '.\ComfyUI\main.py', '--windows-standalone-build', '--use-sage-attention', '--disable-dynamic-vram',
           '--listen', '127.0.0.1', '--port', '8190')
    $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
         -RedirectStandardOutput "$D\comfy.out.log" -RedirectStandardError "$D\comfy.log"
    try {
        foreach ($n in $Prompts) {
            Write-Host "=== $n inicio $(Get-Date -Format T)"
            & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\prompt_${n}_api.json" "$D\amostras_$n.csv"
            Write-Host "=== $n fim $(Get-Date -Format T) rc=$LASTEXITCODE"
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
Write-Host "=== FIM qwen21 $(Get-Date -Format T)"
