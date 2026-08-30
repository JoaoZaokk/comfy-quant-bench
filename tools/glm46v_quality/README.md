# GLM-4.6V local photo reviewer

This Compose file serves the local Huihui GLM-4.6V checkpoint through an
OpenAI-compatible SGLang API on `127.0.0.1:30000`.

The model is restricted to GPU 0 (the RTX 3090). GPU 1 remains available for
ComfyUI. Both model and photo mounts are read-only.

From PowerShell:

```powershell
cd F:\COMFY_PORTABLE\tools\glm46v_quality
# Use a local, read-only copy for reliable Docker access. The NAS source is kept intact.
$env:GLM_NAS_HOST_PATH = 'F:\COMFY_PORTABLE\.scratch\glm46v_model\Huihui-GLM-4.6V-Flash-abliterated'
docker compose -f .\compose.yaml config
docker compose -f .\compose.yaml up -d
docker compose -f .\compose.yaml logs -f glm46v
```

The first start downloads the SGLang image; the model is loaded from the local
copy selected by `GLM_NAS_HOST_PATH`.
The API becomes ready at `http://127.0.0.1:30000/v1`.

`P:` is a mapped SMB/NAS drive on this Windows host. Docker Desktop could list
the share but could not read the model files reliably, so this setup uses a
local read-only copy. Image requests are sent as local base64 data by the
review script; no photo is uploaded to an external service.

This is an evaluation service, not a public endpoint. Keep the loopback port
binding and do not expose it with `0.0.0.0`.
