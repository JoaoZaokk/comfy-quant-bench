Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\encadeia3.log" -Pattern 'FIM encadeia3' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
try { & "$D\roda_fase5.ps1" *> "$D\fase5.log" } catch { Write-Host "ERRO fase5: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_p010.txt' -Dono 'comfy:qwen21_bateria_p010' -Log 'p010_comfy' *> "$D\roda_p010.log" } catch { Write-Host "ERRO p010: $_" }
Write-Host "=== FIM encadeia4 $(Get-Date -Format T)"
