Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\encadeia5.log" -Pattern 'FIM encadeia5' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_dynlocal.txt' -Dono 'comfy:qwen21_runtime_dynlocal' -Log 'rt_dynlocal_comfy' -Dinamico -Extra @('--extra-model-paths-config', "$D\extra_local.yaml") *> "$D\roda_rt_dynlocal.log" } catch { Write-Host "ERRO dynlocal: $_" }
Write-Host "=== FIM encadeia6 $(Get-Date -Format T)"
