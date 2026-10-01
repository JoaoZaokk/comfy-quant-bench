# Sobe o ComfyUI com os argumentos do run_nvidia_gpu_8190_dual_component.bat (legacy UI) para testar a interface.
# Lock + só a 3090; fica no ar até `para_ui_teste.ps1`. PID em ui_teste.pid.
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$s = Get-GpuLockState
if ($s.Owner -ne 'comfy:teste_ui_manager_ctrlz') { Assert-GpuLock -Owner 'comfy:teste_ui_manager_ctrlz' }  # reuso o meu, nunca o de outro
$env:CUDA_VISIBLE_DEVICES = '0'
$env:COMFYUI_MGPU_DISABLED = '1'
$env:PYTHONUTF8 = '1'
$D = '.scratch\triton_2026-09-29'
$p = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList '-s', 'ComfyUI\main.py', '--windows-standalone-build', '--enable-triton-backend', '--use-sage-attention', '--disable-dynamic-vram', '--enable-manager-legacy-ui', '--disable-auto-launch', $env:EXTRA_TESTE, '--listen', '127.0.0.1', '--port', '8190' -RedirectStandardOutput "$D\ui_teste.log" -RedirectStandardError "$D\ui_teste.err" -PassThru -WindowStyle Hidden
$p.Id | Set-Content "$D\ui_teste.pid"
"servidor pid $($p.Id)"
