# Roda sozinho depois que o ladder do Qwen sair, para a placa nao ficar parada entre as minhas
# checagens. A maquina caiu uma vez hoje (Kernel-Power 41, 19:40:10) e reiniciou cinco vezes em
# dois dias, entao cada etapa grava a propria saida e nenhuma depende de a anterior ter terminado
# dentro do mesmo processo.
#
# NAO toma o lock da GPU. As ferramentas que medem tempo tomam sozinhas, e tomar antes faz elas
# recusarem a propria corrida -- ja medido nesta bancada.
$ErrorActionPreference = 'Continue'
Set-Location F:\COMFY_PORTABLE
$env:CUDA_VISIBLE_DEVICES = '0'
$py = '.\python_embeded\python.exe'
$d  = 'ComfyUI\models\diffusion_models\'

# 1. Esperar o ladder. Espera pelo PID, e depois pelo lock sumir -- o processo pode sair um
#    instante antes de o __exit__ apagar o arquivo, e a etapa seguinte nao usa lock mas a de
#    tempo usa.
Write-Host "== esperando o ladder (pid 17248) =="
while (Get-Process -Id 17248 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
Write-Host "ladder saiu em $(Get-Date -Format HH:mm:ss)"
$t = 0
while ((Test-Path F:\GPU_BENCH.lock) -and $t -lt 120) { Start-Sleep -Seconds 5; $t += 5 }
Write-Host "lock livre: $(-not (Test-Path F:\GPU_BENCH.lock))  (esperou ${t}s)"

# 2. Decodificar. Processo separado por necessidade, nao por gosto: o decode dentro do ladder
#    morre com 'NoneType' object has no attribute 'hostbuf_allocate'.
Write-Host "`n== decode =="
& $py -s .\tools\decode_latents.py bench\quality_ladder_qwen_edit\latents `
    --vae qwen_image_vae.safetensors `
    --out bench\quality_ladder_qwen_edit\imagens
Write-Host "DECODE_EXIT=$LASTEXITCODE"

# 3. O kernel de 4 bits roda MESMO nestes dois arquivos? O sidecar registra a conversao, nao a
#    execucao de hoje. Sem esta contagem, "a imagem ficou boa" pode significar que o caminho
#    quantizado nunca foi tomado.
foreach ($m in @('qwen_image_edit_2511_w4a4', 'qwen_image_edit_2511_mixed')) {
    Write-Host "`n== despacho: $m =="
    & $py -s .\tools\probe_quant_dispatch.py "$($d)$m.safetensors" --mode diffusion --forward-only
    Write-Host "DISPATCH_EXIT_$m=$LASTEXITCODE"
}

# 4. O encoder da cadeia, remedido com o cadeado SOLTO. O numero publicado dele e de 01/09 e
#    descreve o caminho dequantizado: a trava so saiu hoje. Sem isto a cadeia do Qwen fica com
#    uma peca medida numa maquina que nao existe mais.
Write-Host "`n== encoder qwen_2.5_vl_7b, cadeado solto =="
& $py -s .\tools\probe_te_cadeado.py qwen_2.5_vl_7b_w4a4_convrot.safetensors `
    --clip-type qwen_image `
    --referencia qwen_2.5_vl_7b.safetensors `
    --controle qwen_2.5_vl_7b.safetensors
Write-Host "TE_EXIT=$LASTEXITCODE"

Write-Host "`nPOS_LADDER_FIM $(Get-Date -Format HH:mm:ss)"
