# Espera a bateria base terminar e roda, em sequencia (cada um com o proprio lock): fase 2, bateria das nossas, shift.
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\roda_bateria.log" -Pattern 'FIM bateria' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
try { & "$D\roda_fase2.ps1" *> "$D\fase2.log" } catch { Write-Host "ERRO fase2: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_nossas.txt' -Dono 'comfy:qwen21_bateria_nossas' -Log 'nossas_comfy' *> "$D\roda_nossas.log" } catch { Write-Host "ERRO nossas: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_shift.txt' -Dono 'comfy:qwen21_shift_2048' -Log 'shift_comfy' *> "$D\roda_shift.log" } catch { Write-Host "ERRO shift: $_" }
Write-Host "=== FIM encadeia $(Get-Date -Format T)"
