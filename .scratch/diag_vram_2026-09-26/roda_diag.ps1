# Duas rodadas frias (servidor reiniciado antes de cada): base e vae1. Lock da GPU do inicio ao fim.
# Revisao 2026-09-29: cada servidor via Invoke-ComfyServer (tools\comfy_server.ps1), sob UM lock tomado
# aqui; o lock so e solto se o ultimo servidor derrubou a porta (antes: sem a correcao 'Continue' de
# 27/09, e o Release rodava mesmo com o servidor de pe).
param([string[]]$Variantes = @('base', 'vae1'))
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\diag_vram_2026-09-26'
. F:\COMFY_PORTABLE\tools\comfy_server.ps1
Assert-GpuLock -Owner 'comfy:diag_vram_eros_2a_passada'
$estado = @{ Caiu = $true; Pid = $null }
try {
    foreach ($v in $Variantes) {
        Write-Host "=== $v inicio $(Get-Date -Format T)"
        # ComfyUI 0.34+ esconde a 3080 Ti no Windows sem CUDA_VISIBLE_DEVICES=0,1 (26/09)
        $a = @('-s', "$D\lanca_comfy_diag.py", "$D\vram_$v.jsonl", '--', '--windows-standalone-build',
               '--use-sage-attention', '--disable-dynamic-vram', '--listen', '127.0.0.1', '--port', '8190')
        if ($v -match 'fp16') { $a += '--fp16-intermediates' }
        Invoke-ComfyServer -Porta 8190 -Estado $estado -Cvd '0,1' -Log "$D\comfy_$v" -ArgsPython $a `
            -PadraoProcesso 'lanca_comfy_diag\.py|main\.py.*--port 8190' -EsperaAposQuedaSeg 10 -Bloco {
            # sem o hook instalado a rodada nao serve (em 26/09 o Manager relancou o main.py com os.execv)
            $j = "$D\vram_$v.jsonl"
            # o processo inicial sai no execv do Manager; o relancado e outro PID, por isso espero pelo arquivo
            for ($i = 0; $i -lt 400 -and -not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet)); $i++) {
                Start-Sleep 2
            }
            if (-not ((Test-Path $j) -and (Select-String -Path $j -Pattern '"instalado"' -Quiet))) { throw "hook de VRAM nao instalou em $v" }
            & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\prompt_${v}_api.json" "$D\amostras_$v.csv" "$D\resultados_diag.jsonl"
        }
        Write-Host "=== $v fim $(Get-Date -Format T)"
        if (-not $estado.Caiu) { Write-Host "porta 8190 nao caiu depois de $v -- parando"; break }
    }
} finally {
    Complete-GpuLock -Estado $estado -Porta 8190
}
Write-Host "=== FIM diag $(Get-Date -Format T)"
