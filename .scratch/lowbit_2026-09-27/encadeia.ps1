# Fases em sequencia; cada uma toma e solta o lock no roda_lowbit.ps1 (que muda o diretorio: caminho absoluto).
$R = 'F:\COMFY_PORTABLE\.scratch\lowbit_2026-09-27\roda_lowbit.ps1'
& $R -Fase offload -Extra @('--novram')
& $R -Fase gpu1
Write-Host "=== FIM ENCADEIA $(Get-Date -Format T)"
