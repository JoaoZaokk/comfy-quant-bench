$ErrorActionPreference = 'Stop'
. F:\COMFY_PORTABLE\tools\comfy_server.ps1
# MOCKS: nunca tocar o lock real
$script:log = New-Object System.Collections.ArrayList
function Assert-GpuLock { param($Owner, [switch]$Force) [void]$script:log.Add("assert:$Owner") }
function Release-GpuLock { [void]$script:log.Add("release") }
$falhas = 0
function Confere($cond, $msg) { if ($cond) { Write-Host "PASS  $msg" } else { Write-Host "FAIL  $msg"; $script:falhas++ } }
$porta = 39571
$arg = @('-s', '-m', 'http.server', "$porta", '--bind', '127.0.0.1')
$env:CUDA_VISIBLE_DEVICES = '-1'

# 1. caminho feliz: sobe, bloco ve a porta, derruba, solta
$script:log.Clear()
Invoke-ComfyUnderLock -Dono 'teste:falso' -Porta $porta -ArgsPython $arg -Cvd '-1' -PadraoProcesso "http\.server $porta" -Bloco {
    for ($i = 0; $i -lt 50 -and -not (Test-ComfyPortaEscutando -Porta $porta); $i++) { Start-Sleep -Milliseconds 200 }
    $script:viu = Test-ComfyPortaEscutando -Porta $porta
}
Confere $script:viu 'bloco viu a porta escutando'
Confere (-not (Test-ComfyPortaEscutando -Porta $porta)) 'porta caiu depois'
Confere (($script:log -join ',') -eq 'assert:teste:falso,release') "lock: $($script:log -join ',')"

# 2. bloco lanca excecao: servidor derrubado e lock solto mesmo assim, excecao propaga
$script:log.Clear(); $propagou = $false
try {
    Invoke-ComfyUnderLock -Dono 'teste:falso' -Porta $porta -ArgsPython $arg -Cvd '-1' -PadraoProcesso "http\.server $porta" -Bloco {
        for ($i = 0; $i -lt 50 -and -not (Test-ComfyPortaEscutando -Porta $porta); $i++) { Start-Sleep -Milliseconds 200 }
        throw 'boom'
    }
} catch { $propagou = ($_.Exception.Message -eq 'boom') }
Confere $propagou 'excecao do bloco propagou'
Confere (-not (Test-ComfyPortaEscutando -Porta $porta)) 'porta caiu apos excecao'
Confere (($script:log -join ',') -eq 'assert:teste:falso,release') "lock apos excecao: $($script:log -join ',')"

# 3. porta ja ocupada antes: recusa, nao mata o ocupante, solta (nada nosso subiu)
$ocupante = Start-Process -FilePath F:\COMFY_PORTABLE\python_embeded\python.exe -ArgumentList $arg -PassThru -NoNewWindow
for ($i = 0; $i -lt 50 -and -not (Test-ComfyPortaEscutando -Porta $porta); $i++) { Start-Sleep -Milliseconds 200 }
$script:log.Clear(); $recusou = $false
try {
    Invoke-ComfyUnderLock -Dono 'teste:falso' -Porta $porta -ArgsPython $arg -Cvd '-1' -PadraoProcesso "http\.server $porta" -Bloco { throw 'nao devia rodar' }
} catch { $recusou = ($_.Exception.Message -match 'ja escuta') }
Confere $recusou 'recusou porta ocupada'
Confere (-not $ocupante.HasExited) 'ocupante continua vivo'
Confere (($script:log -join ',') -eq 'assert:teste:falso,release') "lock porta ocupada: $($script:log -join ',')"
Stop-Process -Id $ocupante.Id -Force   # processo deste teste

# 4. porta que nao cai: lock MANTIDO
$script:log.Clear()
Complete-GpuLock -Estado @{ Caiu = $false; Pid = 1 } -Porta $porta 3>$null
Confere ($script:log.Count -eq 0) 'porta de pe => lock mantido'

Write-Host "falhas: $falhas"
exit $falhas
