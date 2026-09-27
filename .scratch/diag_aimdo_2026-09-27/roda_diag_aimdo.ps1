# Reproduz o abort do dynamic VRAM com o BF16 do Qwen 2.1 lido do NAS, com o read_file_slice instrumentado.
# Variantes: 'repro' (sem conserto) e 'fix' (DIAG_FIX=1). Boot frio por variante, dynamic VRAM ligado, lock do inicio ao fim.
param([string[]]$Variantes = @('repro2', 'sempin'), [string]$Cvd = '0,1', [string]$Porta = '8190', [string]$Grafo = '.scratch\qwen21_2026-09-26\bateria\rt_dyn_bf16_p0_s42.json')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\diag_aimdo_2026-09-27'
. F:\COMFY_PORTABLE\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:diag_aimdo_nas'
try {
    foreach ($v in $Variantes) {
        for ($i = 0; $i -lt 120 -and (Get-NetTCPConnection -LocalPort $Porta -State Listen -EA SilentlyContinue); $i++) { Start-Sleep 5 }
        Write-Host "=== $v inicio $(Get-Date -Format T)"
        $env:COMFYUI_MGPU_DISABLED = '1'
        $env:COMFY_PORT = $Porta
        $env:CUDA_VISIBLE_DEVICES = $Cvd
        $env:DIAG_FIX = '0'
        $env:DIAG_INJETA = if ($v -match 'injeta') { '40' } else { '0' }
        $j = "$D\aimdo_$v.jsonl"
        $a = @('-s', "$D\lanca_diag_aimdo.py", $j, '--', '--windows-standalone-build', '--use-sage-attention',
               '--listen', '127.0.0.1', '--port', $Porta)
        if ($v -match 'sempin') { $a += '--disable-pinned-memory' }
        $p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList $a -PassThru -NoNewWindow `
             -RedirectStandardOutput "$D\comfy_$v.out.log" -RedirectStandardError "$D\comfy_$v.log"
        try {
            for ($i = 0; $i -lt 400 -and -not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet)); $i++) { Start-Sleep 2 }
            if (-not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet))) { throw "instrumentacao nao instalou em $v" }
            & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py $Grafo "$D\amostra_$v.csv"
            Write-Host "=== $v runner rc=$LASTEXITCODE"
        } finally {
            & taskkill.exe /PID $p.Id /T /F 2>$null | Out-Null
            Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                ? { [string]$_.CommandLine -match "lanca_diag_aimdo\.py.*$Porta|main\.py.*--port $Porta" } |
                % { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
            for ($i = 0; $i -lt 60; $i++) { if (-not (Get-NetTCPConnection -LocalPort $Porta -State Listen -EA SilentlyContinue)) { break }; Start-Sleep 2 }
            Start-Sleep 10
        }
        Write-Host "=== $v fim $(Get-Date -Format T)"
    }
} finally {
    Release-GpuLock
}
Write-Host "=== FIM diag_aimdo $(Get-Date -Format T)"
