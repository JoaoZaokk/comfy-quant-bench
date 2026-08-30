@echo off
REM ---------------------------------------------------------------------------
REM  ComfyUI :8190, ouvindo SO em loopback.
REM
REM  Por que este arquivo existe (2026-08-12):
REM  O run_nvidia_gpu_8190.bat usa "--listen 0.0.0.0", e nesta maquina isso faz
REM  o servidor MORRER SEM MORRER: o processo continua vivo (o Manager segue
REM  baixando), mas o accept loop do asyncio cai com
REM
REM      OSError(22, 'The specified network name is no longer available', 64)
REM      Accept failed on a socket -- laddr=('0.0.0.0', 8190)
REM
REM  Um socket em 0.0.0.0 esta preso a TODAS as interfaces. Esta maquina tem
REM  NordLynx (tunel NordVPN) ativo, mais TAP-NordVPN, OpenVPN DCO, Hyper-V
REM  Default Switch e o vEthernet do WSL. Quando o VPN reconecta, a interface e
REM  derrubada e o Windows devolve WSAENETNAME no accept; o asyncio trata como
REM  fatal para aquele socket e para de aceitar conexoes. O ComfyUI fica de pe,
REM  sem porta.
REM
REM  O loopback nunca some. E o app fala com o ComfyUI em 127.0.0.1 de qualquer
REM  forma -- o acesso remoto passa pelo servidor Rust do proprio app, com
REM  tunel, nao por expor a porta do ComfyUI na rede.
REM
REM  Se um dia voce PRECISAR do ComfyUI visivel na LAN, use o .bat original e
REM  saiba que ele cai junto com o VPN.
REM ---------------------------------------------------------------------------
cd /d F:\COMFY_PORTABLE
set CUDA_VISIBLE_DEVICES=0,1
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-sage-attention --listen 127.0.0.1 --port 8190
pause
