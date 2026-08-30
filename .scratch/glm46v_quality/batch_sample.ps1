$root = 'C:\Users\joaoz\Downloads\Photos-1-001\sku\fotos-anuncios-2026-08-26'
$all = @(Get-ChildItem -LiteralPath $root -Filter *.jpg -File -Recurse)
$pick = @(
    $all | Sort-Object Length -Descending | Select-Object -First 3
    $all | Where-Object { $_.Length -lt 200000 } | Sort-Object Length | Select-Object -First 2
) | Sort-Object FullName -Unique
$out = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'sample_results.jsonl'
Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
foreach ($img in $pick) {
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($img.FullName))
    $content = @(
        @{ type = 'image_url'; image_url = @{ url = ('data:image/jpeg;base64,' + $b64) } }
        @{ type = 'text'; text = 'Avalie esta foto de produto para anúncio. Responda em JSON válido e curto com: qualidade_tecnica_0a10, centralizacao_0a10, fundo_0a10, legibilidade_texto_0a10, problemas, aprovada. Não invente detalhes.' }
    )
    $body = @{
        model = 'glm-4.6v'
        messages = @(@{ role = 'user'; content = $content })
        max_tokens = 256
        temperature = 0.0
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 8
    $sw = [Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:30000/v1/chat/completions' -ContentType 'application/json' -Body $body -TimeoutSec 180
        $sw.Stop()
        $record = [ordered]@{ file = $img.FullName; bytes = $img.Length; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); result = $resp.choices[0].message.content }
    } catch {
        $sw.Stop()
        $record = [ordered]@{ file = $img.FullName; bytes = $img.Length; seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); error = $_.Exception.Message }
    }
    $record | ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath $out -Encoding utf8
    Write-Output ($record | ConvertTo-Json -Compress -Depth 6)
}
Write-Output ('RESULT_FILE=' + $out)
