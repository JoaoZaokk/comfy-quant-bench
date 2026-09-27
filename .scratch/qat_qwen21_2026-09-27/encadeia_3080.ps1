# Na 3080 Ti (a 3090 esta presa por um zumbi), porta 8191: sem-pin, bateria QAT x RTN, conserto do aimdo.
Set-Location F:\COMFY_PORTABLE
while (-not (Select-String -Path '.scratch\diag_aimdo_2026-09-27\roda_sempin.log' -Pattern 'FIM diag_aimdo' -Quiet)) { Start-Sleep 20 }
Start-Sleep 10
try { & .scratch\diag_aimdo_2026-09-27\roda_diag_aimdo.ps1 -Variantes @('sempin_3080b') -Cvd '1' -Porta '8191' *> .scratch\diag_aimdo_2026-09-27\roda_sempin_b.log } catch { Write-Host "ERRO sempin: $_" }
try { & .scratch\qwen21_2026-09-26\roda_bateria.ps1 -Ordem 'bateria\ordem_qat.txt' -Dono 'comfy:qwen21_bateria_qat' -Log 'qat_comfy' -Porta '8191' -Cvd '1' *> .scratch\qwen21_2026-09-26\roda_qat.log } catch { Write-Host "ERRO qat: $_" }
try { & .scratch\diag_aimdo_2026-09-27\roda_diag_aimdo.ps1 -Variantes @('fixcomfy_3080') -Cvd '1' -Porta '8191' *> .scratch\diag_aimdo_2026-09-27\roda_fix.log } catch { Write-Host "ERRO fix: $_" }
Write-Host "=== FIM encadeia_3080 $(Get-Date -Format T)"
