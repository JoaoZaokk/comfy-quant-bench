# Depois da segunda cadeia: fase 4 (escalas refinadas) e bateria dos refinados.
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\encadeia2.log" -Pattern 'FIM encadeia2' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
try { & "$D\roda_fase4.ps1" *> "$D\fase4.log" } catch { Write-Host "ERRO fase4: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_escalas.txt' -Dono 'comfy:qwen21_bateria_escalas' -Log 'escalas_comfy' *> "$D\roda_escalas.log" } catch { Write-Host "ERRO escalas: $_" }
Write-Host "=== FIM encadeia3 $(Get-Date -Format T)"
