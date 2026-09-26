# Duas rodadas frias (servidor reiniciado antes de cada): base e vae1. Lock da GPU do inicio ao fim.
param([string[]]$Variantes = @('base', 'vae1'))
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\diag_vram_2026-09-26'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:diag_vram_eros_2a_passada'
try {
    foreach ($v in $Variantes) {
        Write-Host "=== $v inicio $(Get-Date -Format T)"
        $env:COMFYUI_MGPU_DISABLED = '1'
        $a = @('-s', "$D\lanca_comfy_diag.py", "$D\vram_$v.jsonl", '--', '--windows-standalone-build',
               '--use-sage-attention', '--disable-dynamic-vram', '--listen', '127.0.0.1', '--port', '8190')
        if ($v -match 'fp16') { $a += '--fp16-intermediates' }
        $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
             -RedirectStandardOutput "$D\comfy_$v.out.log" -RedirectStandardError "$D\comfy_$v.log"
        try {
            # sem o hook instalado a rodada nao serve (em 26/09 o Manager relancou o main.py com os.execv)
            $j = "$D\vram_$v.jsonl"
            # o processo inicial sai no execv do Manager; o relancado e outro PID, por isso espero pelo arquivo
            for ($i = 0; $i -lt 400 -and -not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet)); $i++) {
                Start-Sleep 2
            }
            if (-not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet))) { throw "hook de VRAM nao instalou em $v" }
            & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\prompt_${v}_api.json" "$D\amostras_$v.csv"
        } finally {
            & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                ? { [string]$_.CommandLine -match 'lanca_comfy_diag\.py|main\.py.*--port 8190' } |
                % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
            for ($i = 0; $i -lt 60; $i++) {
                if (-not (Get-NetTCPConnection -LocalPort 8190 -State Listen -EA SilentlyContinue)) { break }
                Start-Sleep 2
            }
            Start-Sleep 10
        }
        Write-Host "=== $v fim $(Get-Date -Format T)"
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM diag $(Get-Date -Format T)"
