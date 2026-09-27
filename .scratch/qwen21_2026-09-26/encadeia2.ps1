# Depois da primeira cadeia: fase 3 (quants FP32), bateria das nossas, runtime (dynamic VRAM, sem os commits da master).
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\encadeia.log" -Pattern 'FIM encadeia' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
try { & "$D\roda_fase3.ps1" *> "$D\fase3.log" } catch { Write-Host "ERRO fase3: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_nossas_f32.txt' -Dono 'comfy:qwen21_bateria_nossas' -Log 'nossas_comfy' *> "$D\roda_nossas.log" } catch { Write-Host "ERRO nossas: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_dyn.txt' -Dono 'comfy:qwen21_runtime_dyn' -Log 'rt_dyn_comfy' -Dinamico *> "$D\roda_rt_dyn.log" } catch { Write-Host "ERRO rt_dyn: $_" }
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_base.txt' -Dono 'comfy:qwen21_runtime_base' -Log 'rt_base_comfy' *> "$D\roda_rt_base.log" } catch { Write-Host "ERRO rt_base: $_" }
Write-Host "=== FIM encadeia2 $(Get-Date -Format T)"
