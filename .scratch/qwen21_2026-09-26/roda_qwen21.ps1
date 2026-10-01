# Um boot do ComfyUI 0.37.4 e os prompts do Qwen-Image-2.1 em sequencia. Lock da GPU do inicio ao fim.
# Revisao 2026-09-29: via tools\comfy_server.ps1 -- esta copia nao tinha a correcao 'Continue' de 27/09
# e podia soltar o lock com o servidor ainda vivo.
param([string[]]$Prompts = @('int8_s42', 'mixed_s42', 'int4_s42'))
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
$D = '.scratch\qwen21_2026-09-26'
. F:\COMFY_PORTABLE\tools\comfy_server.ps1
Invoke-ComfyUnderLock -Dono 'comfy:qwen21_primeiro_render' -Porta 8190 -Cvd '0,1' -Log "$D\comfy" `
    -ArgsComfy @('--use-sage-attention', '--disable-dynamic-vram') -Bloco {
    foreach ($n in $Prompts) {
        Write-Host "=== $n inicio $(Get-Date -Format T)"
        & .\python_embeded\python.exe -s .scratch\roda_eros_2gpu.py "$D\prompt_${n}_api.json" "$D\amostras_$n.csv" "$D\resultados_qwen21.jsonl"
        Write-Host "=== $n fim $(Get-Date -Format T) rc=$LASTEXITCODE"
    }
}
Write-Host "=== FIM qwen21 $(Get-Date -Format T)"
