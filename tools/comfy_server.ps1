# Sobe um ComfyUI, roda um bloco contra ele, derruba -- e so solta o lock da GPU depois de CONFIRMAR
# que a porta caiu. Dot-source:
#
#     . F:\COMFY_PORTABLE\tools\comfy_server.ps1
#     Invoke-ComfyUnderLock -Dono 'comfy:minha_bateria' -Porta 8190 -Log "$D\bateria_comfy" `
#         -ArgsComfy @('--use-sage-attention', '--disable-dynamic-vram') -Bloco {
#             & .\python_embeded\python.exe -s tools\comfy_client.py --server 127.0.0.1:8190 `
#                 --lista "$D\bateria\ordem.txt" --saida "$D\bateria\resultados.jsonl"
#         }
#
# POR QUE EXISTE (revisao 2026-09-29, achado 3). Havia quatro copias deste roteiro em .scratch
# (roda_bateria, roda_lowbit, roda_qwen21, roda_diag). A correcao de 27/09 -- no Windows
# PowerShell 5, com $ErrorActionPreference='Stop', o stderr do taskkill ("not found") vira erro
# terminante e PULA a segunda limpeza, deixando o servidor vivo -- so entrou em duas. Nas outras,
# o `finally` externo rodava `Release-GpuLock` com o ComfyUI possivelmente vivo segurando VRAM:
# lock solto com trabalho de pe, o que o protocolo proibe ("soltar so apos o trabalho parar").
#
# O QUE ESTE ARQUIVO GARANTE, e so isto:
#   - recusa subir se a porta JA escuta antes (o processo ali nao e nosso, e a limpeza por linha de
#     comando mataria o de outra pessoa);
#   - derruba com $ErrorActionPreference='Continue' (a correcao de 27/09, num lugar so);
#   - `Release-GpuLock` so quando a porta parou de escutar dentro do prazo, ou quando o servidor
#     nunca chegou a subir. Se a porta nao cai, o lock FICA, com aviso -- um lock preso e a falha
#     barata; um lock solto com a placa ocupada corrompe a medicao de outro.
# Protocolo do lock inalterado: usa Assert-GpuLock / Release-GpuLock de gpu_lock.ps1.

. "$PSScriptRoot\gpu_lock.ps1"

function Test-ComfyPortaEscutando {
    param([Parameter(Mandatory)][int]$Porta)
    [bool](Get-NetTCPConnection -LocalPort $Porta -State Listen -EA SilentlyContinue)
}

function Stop-ComfyServer {
    # Derruba o processo que subimos e os filhos/relancados que casam com $PadraoProcesso (o
    # Manager relanca main.py por os.execv com outro PID). Devolve $true se a porta caiu no prazo.
    param([Parameter(Mandatory)]$Processo, [Parameter(Mandatory)][int]$Porta,
          [Parameter(Mandatory)][string]$PadraoProcesso, [int]$PrazoSeg = 120)
    $ErrorActionPreference = 'Continue'   # escopo desta funcao: stderr do taskkill nao aborta nada
    & taskkill.exe /PID $Processo.Id /T /F 2>$null | Out-Null
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
        Where-Object { [string]$_.CommandLine -match $PadraoProcesso } |
        ForEach-Object { & taskkill.exe /PID $_.ProcessId /T /F 2>$null | Out-Null }
    $limite = [DateTime]::Now.AddSeconds($PrazoSeg)
    while ([DateTime]::Now -lt $limite) {
        if (-not (Test-ComfyPortaEscutando -Porta $Porta)) { return $true }
        Start-Sleep -Seconds 2
    }
    return (-not (Test-ComfyPortaEscutando -Porta $Porta))
}

function Invoke-ComfyServer {
    # Sobe, roda $Bloco, derruba. Sem lock: quem chama e dono do lock (Invoke-ComfyUnderLock, ou
    # um script que roda varios servidores frios sob um lock so). `$Estado.Caiu` diz, mesmo quando
    # o bloco lanca excecao, se e seguro soltar o lock.
    param(
        [Parameter(Mandatory)][int]$Porta,
        [Parameter(Mandatory)][scriptblock]$Bloco,
        [Parameter(Mandatory)][hashtable]$Estado,
        [string]$Log = '',
        [string[]]$ArgsComfy = @(),
        [string[]]$ArgsPython = @(),        # substitui o main.py inteiro (ex.: um lancador de diagnostico)
        [string]$Cvd = '0,1',
        [hashtable]$Ambiente = @{},
        [string]$PadraoProcesso = '',
        [int]$PrazoQuedaSeg = 120,
        [int]$EsperaAposQuedaSeg = 0,
        [string]$Raiz = 'F:\COMFY_PORTABLE'
    )
    $Estado.Caiu = $true                    # nada subiu ainda: soltar e seguro
    if (-not $PadraoProcesso) { $PadraoProcesso = "main\.py.*--port $Porta(\s|$)" }
    if (Test-ComfyPortaEscutando -Porta $Porta) {
        throw "porta $Porta ja escuta antes de subir -- o processo ali nao e deste script; recusando"
    }
    if (-not $ArgsPython -or $ArgsPython.Count -eq 0) {
        $ArgsPython = @('-s', '.\ComfyUI\main.py', '--windows-standalone-build',
                        '--listen', '127.0.0.1', '--port', "$Porta") + $ArgsComfy
    }
    $env:COMFYUI_MGPU_DISABLED = '1'
    $env:COMFY_PORT = "$Porta"
    $env:CUDA_VISIBLE_DEVICES = $Cvd
    foreach ($k in $Ambiente.Keys) { Set-Item -Path "env:$k" -Value $Ambiente[$k] }
    $sp = @{ FilePath = (Join-Path $Raiz 'python_embeded\python.exe'); ArgumentList = $ArgsPython
             PassThru = $true; NoNewWindow = $true; WorkingDirectory = $Raiz }
    if ($Log) { $sp.RedirectStandardOutput = "$Log.out.log"; $sp.RedirectStandardError = "$Log.log" }
    $p = Start-Process @sp
    $Estado.Caiu = $false                   # a partir daqui ha um servidor nosso de pe
    $Estado.Pid = $p.Id
    try {
        & $Bloco
    } finally {
        $Estado.Caiu = Stop-ComfyServer -Processo $p -Porta $Porta -PadraoProcesso $PadraoProcesso `
                                        -PrazoSeg $PrazoQuedaSeg
        if ($Estado.Caiu -and $EsperaAposQuedaSeg -gt 0) { Start-Sleep -Seconds $EsperaAposQuedaSeg }
    }
}

function Complete-GpuLock {
    # Solta o lock so se o servidor caiu (ou nunca subiu). Senao, mantem e diz por que.
    param([Parameter(Mandatory)][hashtable]$Estado, [int]$Porta = 0)
    if ($Estado.Caiu) {
        Release-GpuLock
    } else {
        Write-Warning ("porta $Porta ainda escuta (pid inicial $($Estado.Pid)): lock da GPU MANTIDO. " +
                       "Confira o processo e so entao rode Release-GpuLock a mao.")
    }
}

function Invoke-ComfyUnderLock {
    # Assert-GpuLock -> Invoke-ComfyServer -> Release-GpuLock somente com a porta caida.
    param(
        [Parameter(Mandatory)][string]$Dono,
        [Parameter(Mandatory)][int]$Porta,
        [Parameter(Mandatory)][scriptblock]$Bloco,
        [string]$Log = '',
        [string[]]$ArgsComfy = @(),
        [string[]]$ArgsPython = @(),
        [string]$Cvd = '0,1',
        [hashtable]$Ambiente = @{},
        [string]$PadraoProcesso = '',
        [int]$PrazoQuedaSeg = 120,
        [int]$EsperaAposQuedaSeg = 0
    )
    Assert-GpuLock -Owner $Dono
    $estado = @{ Caiu = $true; Pid = $null }
    try {
        Invoke-ComfyServer -Porta $Porta -Bloco $Bloco -Estado $estado -Log $Log -ArgsComfy $ArgsComfy `
            -ArgsPython $ArgsPython -Cvd $Cvd -Ambiente $Ambiente -PadraoProcesso $PadraoProcesso `
            -PrazoQuedaSeg $PrazoQuedaSeg -EsperaAposQuedaSeg $EsperaAposQuedaSeg
    } finally {
        Complete-GpuLock -Estado $estado -Porta $Porta
    }
}
