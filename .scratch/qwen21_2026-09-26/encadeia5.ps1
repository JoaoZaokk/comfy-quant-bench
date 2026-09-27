# Isolamento do crash do dynamic VRAM e commits da master 2f7c6d47 + 1d61dcc3 (aplicados sem commit, revertidos no fim).
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
while (-not (Select-String -Path "$D\encadeia4.log" -Pattern 'FIM encadeia4' -Quiet)) { Start-Sleep 20 }
Start-Sleep 15
$ARQ = @('comfy/ldm/qwen_image21/model.py', 'comfy/ldm/wan/model_animate2.py', 'comfy/model_base.py', 'comfy/model_management.py')
try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_dyn2.txt' -Dono 'comfy:qwen21_runtime_dyn2' -Log 'rt_dyn2_comfy' -Dinamico *> "$D\roda_rt_dyn2.log" } catch { Write-Host "ERRO dyn2: $_" }
$sujo = & git -C ComfyUI status --porcelain -- @ARQ
if ($sujo) { Write-Host "ERRO: arquivos alvo ja modificados, nao aplico os commits: $sujo" }
else {
    & git -C ComfyUI cherry-pick -n 2f7c6d47 1d61dcc3
    if ($LASTEXITCODE -ne 0) { Write-Host "ERRO cherry-pick rc=$LASTEXITCODE"; & git -C ComfyUI checkout HEAD -- @ARQ }
    else {
        Write-Host "=== commits da master aplicados $(Get-Date -Format T)"
        try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_dynm.txt' -Dono 'comfy:qwen21_runtime_dynm' -Log 'rt_dynm_comfy' -Dinamico *> "$D\roda_rt_dynm.log" } catch { Write-Host "ERRO dynm: $_" }
        try { & "$D\roda_bateria.ps1" -Ordem 'bateria\ordem_rt_base_mestre.txt' -Dono 'comfy:qwen21_runtime_base_mestre' -Log 'rt_base_mestre_comfy' *> "$D\roda_rt_base_mestre.log" } catch { Write-Host "ERRO base_mestre: $_" }
        & git -C ComfyUI reset -q HEAD -- @ARQ
        & git -C ComfyUI checkout HEAD -- @ARQ
        $resto = & git -C ComfyUI status --porcelain -- @ARQ
        Write-Host "=== commits revertidos; resto: [$resto] $(Get-Date -Format T)"
    }
}
Write-Host "=== FIM encadeia5 $(Get-Date -Format T)"
