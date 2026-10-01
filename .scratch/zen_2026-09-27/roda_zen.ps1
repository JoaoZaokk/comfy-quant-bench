# Lock da GPU do inicio ao fim: ComfyUI de teste (transformers 5 isolado, so o no zen) e a bateria de 12.
param([string]$Porta = '8192')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\zen_2026-09-27'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:zen_image_edit_te'
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:HF_HUB_OFFLINE = '1'
    $env:COMFYUI_MGPU_DISABLED = '1'
    $a = @('-s', "$D\main_tf5.py", '--windows-standalone-build', '--use-sage-attention', '--disable-dynamic-vram',
           '--listen', '127.0.0.1', '--port', $Porta, '--extra-model-paths-config', "$D\extra_zen.yaml",
           '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'zen-image-edit-comfyui')
    $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
         -RedirectStandardOutput "$D\comfy_zen.out.log" -RedirectStandardError "$D\comfy_zen.log"
    try {
        & .\python_embeded\python.exe -s "$D\roda_zen.py" $Porta
        if ($LASTEXITCODE -ne 0) { Write-Host "ERRO rc=$LASTEXITCODE" }
    } finally {
        $ErrorActionPreference = 'Continue'
        & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            ? { [string]$_.CommandLine -match "main_tf5\.py.*--port $Porta" } |
            % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM zen $(Get-Date -Format T)"
