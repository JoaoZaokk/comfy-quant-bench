# Prova que COMFYUI_MGPU_WORKER_FLAGS chega ao worker do multigpu-orchestrator: só a 3090, porta 8199, sob lock.
$ErrorActionPreference = 'Stop'
Set-Location F:\COMFY_PORTABLE
. .\tools\gpu_lock.ps1
$D = '.scratch\triton_2026-09-29'
Assert-GpuLock -Owner 'comfy:confere_worker_flags'
$srv = $null
try {
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:COMFYUI_MGPU_WORKER_FLAGS = '--use-sage-attention --enable-triton-backend'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:COMFYUI_MGPU_DISABLED -ErrorAction SilentlyContinue
    & .\python_embeded\python.exe -s -c "import sys,json;sys.path.insert(0,r'$D');sys.argv=['x','on'];import valida_triton as v;g=v.klein(False,31);g['9']['inputs']['filename_prefix']='triton_2026-09-29/worker/K1';open(r'$D\k1_worker.json','w').write(json.dumps(g))"
    $srv = Start-Process -FilePath .\python_embeded\python.exe -ArgumentList '-s', 'ComfyUI\main.py', '--windows-standalone-build', '--enable-triton-backend', '--use-sage-attention', '--disable-pinned-memory', '--listen', '127.0.0.1', '--port', '8199' -RedirectStandardOutput "$D\worker_main.log" -RedirectStandardError "$D\worker_main.err" -PassThru -WindowStyle Hidden
    & .\python_embeded\python.exe -s tools\comfy_client.py --server 127.0.0.1:8199 --api-prompt "$D\k1_worker.json" --saida "$D\worker_resultado.jsonl" --wait-server 600 --timeout 900
    Write-Host "comfy_client rc=$LASTEXITCODE"
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '--cuda-device' } | ForEach-Object { "WORKER: $($_.CommandLine)" }
    Get-Content ComfyUI\logs\mgpu-workers\gpu-0.log | Select-String 'triton|Enabling comfy-kitchen|sage' | Select-Object -First 5 | ForEach-Object { "LOG: $_" }
} finally {
    $ErrorActionPreference = 'Continue'
    if ($srv) { taskkill /PID $srv.Id /T /F | Out-Null; Start-Sleep 3 }
    Release-GpuLock
}
