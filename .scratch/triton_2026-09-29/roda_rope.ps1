$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
Assert-GpuLock -Owner 'comfy:triton_rope_fp32'
try { $env:CUDA_VISIBLE_DEVICES = '0'; & .\python_embeded\python.exe -s .scratch\triton_2026-09-29\rope_fp32.py }
finally { $ErrorActionPreference = 'Continue'; Release-GpuLock }
