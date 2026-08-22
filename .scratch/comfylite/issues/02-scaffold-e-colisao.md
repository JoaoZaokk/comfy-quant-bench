# 02 - Scaffold ComfyLite beside ComfyUI without touching the bench

Type: task
Status: ready-for-agent
Blocked by: 01
Provenance: EXECUTED for git and ports; OBSERVED for the toolchain

## The toolchain is already complete. Nothing to install for the Rust half.

Measured 2026-08-22 on this host:

| piece | state |
|---|---|
| `rustc` / `cargo` | **1.96.0 (2026-05-25)**, `stable-x86_64-pc-windows-msvc` default |
| `cargo-tauri` | **2.11.2**, already in `~/.cargo/bin` |
| MSVC | four installs (VS 18 Community, VS 18 BuildTools, VS 2022 BT, VS 2019 BT) |
| WebView2 runtime | `151.0.4129.93` present |
| Node | **v22.22.2** via nvm4w; `npm` and `bun` on PATH |
| pnpm | **absent** -- Svelte/Vite run on npm or bun as-is |

`F:/cortiq-cmf` already compiles a nontrivial async Rust web stack (axum 0.8 + tower-http + tokio,
edition 2024, `lto = "thin"`) on the same toolchain, so this is not a cold start.

Python side, `python_embeded/Lib/site-packages`: `aiohttp 3.14.3` (what ComfyUI itself runs on),
`pydantic 2.13.4`, `httpx 0.28.1`, `requests`, `pyyaml 6.0.3`, `huggingface_hub 0.36.2`,
`safetensors 0.8.0`, `psutil`, `nvidia_ml_py`. **`sqlite3` works -- `_sqlite3.pyd` + `sqlite3.dll` are
present and `sqlite_version` is 3.50.4** (executed). **Absent**: fastapi, uvicorn, starlette, flask,
tornado, gradio, websockets, pywebview -- and **tkinter**, which the handoff wanted to avoid anyway and
which is not merely discouraged here but genuinely missing (`_tkinter.pyd` absent, zero `tcl*/tk*`
DLLs; the only hits are Pillow's probe and an MSVC import stub).

**So the Python worker needs zero new dependencies.** `aiohttp.web` gives REST and WebSocket both.
Adding FastAPI would be a package change, which is the owner's call under CLAUDE.md's hard rules -- and
it is avoidable.

## Git: ComfyLite is invisible by default, and that is the trap

Executed 2026-08-22:

```
$ git check-ignore -v ComfyLite/
.gitignore:5:/*    ComfyLite/
```

The root `.gitignore` is an **allowlist**: `/*` ignores everything and paths are re-included one by
one. So `ComfyLite/` is ignored, `git status` stays clean, and git does not descend into it. That is
the "sem afetar nada" property, and it is also exactly the failure `.gitignore:45-47` already records:

> The local issue tracker. Ignored for four days, which meant 26 tickets and the map existed only on
> disk -- one `git clean -xdf` from gone.

**ComfyLite gets its own repo** (owner's decision, 2026-08-22), so it is not re-included in the root
allowlist -- but its own `.git` must exist from the first commit, not after the first week.

`.gitattributes` at the root has rules for `.py`, `.md`, `.json`, `.yaml`, `.sh`, `.bat`, `.ps1` and
binaries -- and **none for `.rs`, `.svelte`, `.ts`, `.js`, `.toml`, `.css`, `.html`.` ComfyLite's own
`.gitattributes` must cover them or they fall to `* text=auto` and normalise differently per editor.
The root file's header records why this is not pedantry: a sibling project lost two benchmark rounds
because `set -euo pipefail` picked up a CR.

## Ports

Claimed on this bench: **8188** (ComfyUI default), **8190** (four launchers + three client defaults),
**8123** (`ltx_studio.py`), **8080** (`cortiq serve`).

Actually listening right now (executed, `Get-NetTCPConnection -State Listen`): a cluster at
**8088-9898** -- 8088, 8090, 8300, 8310, 8399, 8787, 8801, 9000, 9010, 9012-9015, 9090, 9166, 9167,
9210, 9247, 9301, 9898 -- most of it owned by a handful of PIDs, consistent with the owner's ERP stack.

**Pick 8210-8250, bind 127.0.0.1, and make it configurable.** The configurable part is not cosmetic:
`run_nvidia_gpu_8190_loopback.bat:6-11` records that binding `0.0.0.0` on this host makes the asyncio
accept loop die with `OSError(22, 'The specified network name is no longer available', 64)` whenever
NordLynx/NordVPN reconnects -- process alive, port dead.

## Disk

Executed 2026-08-22: **C: 61.5 GB free**, F: 570.9 GB, D: 845.2 GB. The tight drive is C:, which is
where `~/.cargo/registry`, `node_modules` caches and a Rust `target/` land by default.

**Keep `ComfyLite/target/` and `ComfyLite/node_modules/` on F:**, and put both in ComfyLite's own
`.gitignore` on day one.

## Closing criterion

Closed when `F:/COMFY_PORTABLE/ComfyLite/` exists with: its own `.git` and a first commit; a
`.gitattributes` covering `.rs`/`.svelte`/`.ts`/`.toml`/`.css`/`.html`; a `.gitignore` covering
`target/`, `node_modules/`, `__pycache__/`, and the catalog DB; a Tauri 2 shell that builds and opens a
window; and a chosen port in 8210-8250 read from config, bound to loopback. And
`git -C F:/COMFY_PORTABLE status --short` still shows nothing for `ComfyLite/`.
