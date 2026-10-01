# Um boot do ComfyUI e os grafos de bateria/ordem.txt em sequencia (agrupados por DiT). Lock da GPU do inicio ao fim.
# Revisao 2026-09-29: subida/derrubada/lock via tools\comfy_server.ps1 (Invoke-ComfyUnderLock), que so
# solta o lock depois de confirmar a porta caida; cada grafo grava um registro por prompt_id em
# bateria\resultados_<ordem>.jsonl (metricas_bateria.py le esse arquivo direto).
param([string]$Ordem = 'bateria\ordem.txt', [string]$Dono = 'comfy:qwen21_bateria_metricas', [string]$Log = 'bateria_comfy', [switch]$Dinamico, [string[]]$Extra = @(), [string]$Porta = '8190', [string]$Cvd = '0,1')
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
. F:\COMFY_PORTABLE\tools\comfy_server.ps1
$argsComfy = @('--use-sage-attention')
if (-not $Dinamico) { $argsComfy += '--disable-dynamic-vram' }  # -Dinamico: como o .bat do dono
$argsComfy += $Extra
$resultados = "$D\bateria\resultados_$([IO.Path]::GetFileNameWithoutExtension($Ordem)).jsonl"
Invoke-ComfyUnderLock -Dono $Dono -Porta ([int]$Porta) -Cvd $Cvd -Log "$D\$Log" -ArgsComfy $argsComfy -Bloco {
    foreach ($g in Get-Content "$D\$Ordem") {
        if (-not $g) { continue }
        Write-Host "=== $g $(Get-Date -Format T)"
        & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\$g" "$D\bateria\amostra_$([IO.Path]::GetFileNameWithoutExtension($g)).csv" $resultados
        if ($LASTEXITCODE -ne 0) { Write-Host "ERRO rc=$LASTEXITCODE em $g" }
    }
}
Write-Host "=== FIM bateria $(Get-Date -Format T)"
