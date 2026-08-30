$img = 'C:\Users\joaoz\Downloads\Photos-1-001\sku\fotos-anuncios-2026-08-26\lu3173t\LU3173PK\catalogo_06.jpg'
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($img))
$content = @(
    @{ type = 'image_url'; image_url = @{ url = ('data:image/jpeg;base64,' + $b64) } }
    @{ type = 'text'; text = 'Analise esta foto de produto com muito detalhe. Descreva composição, materiais visíveis, cores, iluminação, possíveis problemas técnicos e adequação para anúncio. Escreva uma análise longa e contínua, sem resumir.' }
)
$body = @{
    model = 'glm-4.6v'
    messages = @(@{ role = 'user'; content = $content })
    max_tokens = 2000
    temperature = 0.2
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 8
$sw = [Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:30000/v1/chat/completions' -ContentType 'application/json' -Body $body -TimeoutSec 180
$sw.Stop()
Write-Output ('ELAPSED_SEC=' + [math]::Round($sw.Elapsed.TotalSeconds, 1))
Write-Output ('OUTPUT_CHARS=' + $resp.choices[0].message.content.Length)
