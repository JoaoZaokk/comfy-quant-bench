$img = 'C:\Users\joaoz\Downloads\Photos-1-001\sku\fotos-anuncios-2026-08-26\lu3173t\LU3173PK\catalogo_06.jpg'
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($img))
$content = @(
    @{ type = 'image_url'; image_url = @{ url = ('data:image/jpeg;base64,' + $b64) } }
    @{ type = 'text'; text = 'Avalie esta foto de produto para anúncio. Responda em JSON válido e curto com: produto, qualidade_tecnica_0a10, centralizacao_0a10, fundo_0a10, legibilidade_texto_0a10, problemas, aprovada. Não invente detalhes que não estejam visíveis.' }
)
$bodyObject = @{
    model = 'glm-4.6v'
    messages = @(@{ role = 'user'; content = $content })
    max_tokens = 300
    temperature = 0.0
    chat_template_kwargs = @{ enable_thinking = $false }
}
$body = $bodyObject | ConvertTo-Json -Depth 8
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:30000/v1/chat/completions' -ContentType 'application/json' -Body $body -TimeoutSec 180
    $sw.Stop()
    Write-Output ('ELAPSED_SEC=' + [math]::Round($sw.Elapsed.TotalSeconds, 1))
    $resp.choices[0].message.content
} catch {
    $sw.Stop()
    Write-Output ('ELAPSED_SEC=' + [math]::Round($sw.Elapsed.TotalSeconds, 1))
    Write-Output $_
    exit 1
}
