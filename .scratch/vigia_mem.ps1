$L='F:\COMFY_PORTABLE\bench\quality_ladder_qwen_edit\latents'
1..40 | ForEach-Object {
  $py=Get-Process -Id 17248 -EA SilentlyContinue
  $cm=(Get-Counter '\Memory\Committed Bytes').CounterSamples[0].CookedValue
  $os=Get-CimInstance Win32_OperatingSystem
  $n=(Get-ChildItem $L -Filter *.pt -EA SilentlyContinue|Measure-Object).Count
  "{0} lat={1,2} py_priv={2,6:N2}G py_ws={3,6:N2}G commit={4,6:N2}G livre={5,5:N1}G vivo={6}" -f `
    (Get-Date -Format HH:mm:ss),$n,$(if($py){$py.PrivateMemorySize64/1GB}else{0}),`
    $(if($py){$py.WorkingSet64/1GB}else{0}),($cm/1GB),($os.FreePhysicalMemory/1MB),[bool]$py
  Start-Sleep -Seconds 30
}
