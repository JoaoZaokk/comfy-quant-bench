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

## STATUS 2026-08-22: six of the eight clauses met. NOT closed, and the two gaps are both the owner's.

Checked one clause at a time, EXECUTED:

| clause | state |
|---|---|
| `ComfyLite/` exists with its own `.git` | **yes** -- branch `main` |
| a first commit | **yes** -- `2ec234f`, 55 tracked files, made by the owner 2026-08-23 |
| `.gitattributes` covering `.rs`/`.svelte`/`.ts`/`.toml`/`.css`/`.html` | **yes**, plus `.py`/`.md` and CRLF for `.bat`/`.ps1` |
| `.gitignore` covering `target/`, `node_modules/`, `__pycache__/`, catalog DB | **yes**, plus `src-tauri/gen/schemas/` which tauri-build regenerates |
| a Tauri 2 shell that **builds and opens a window** | **NO** -- see below |
| a port in 8210-8250 read from config, bound to loopback | **yes** -- 8221 in `config.toml`, EXECUTED: bound, served, released |
| root `git status --short` shows nothing for `ComfyLite/` | **yes** -- `.gitignore:5:/*` still hides it |

**Gap 1 -- CLOSED by the owner, 2026-08-23.** `2ec234f "ComfyLite phase 0-2: worker, inventory,
shell"`, 55 tracked files. The project is no longer one `git clean -xdf` from gone.

**Gap 2 -- the window was never opened.** `cargo check` ran and `src-tauri/target/debug` exists with a
4,493-line `Cargo.lock`, so the crate graph resolves and the Rust half compiles. But a *window* needs
`cargo tauri dev`, which is a full link, and `~/.cargo` lives on **C: with 55.4 GiB free** (EXECUTED
via the hardware probe, and lower than the 61.5 GiB this ticket recorded earlier the same day). That
is a disk-space decision, so it waits.

What *was* proved instead, EXECUTED: the Vite build produces `index-CrSa0E5T.js` **94,766 bytes** plus
**24,717 bytes** of CSS, `npm run dev` served on `127.0.0.1:5173`, and the three screens rendered live
data from the real worker -- 505 files, 1.01 TiB, both GPUs, all eleven drives with the four network
mounts labelled. So the frontend is not merely "written"; it is running. It has never been inside a
Tauri window.

**One process hazard found by doing this, worth carrying:** stopping `npm run dev` left `vite.js`
alive as an orphan still holding 5173 -- `npm` dies, its child does not. Same shape as
`main.py --windows-standalone-build` re-executing itself. Kill the tree, not the pid.

## Follow-up, 2026-08-23: two terminals was the wrong shape, and it is fixed

The owner's words were *"abrir dois PS pra poder abrir 1 app e foda"*, plus `http://127.0.0.1:8221/`
answering `404: Not Found` in a browser. Both were real and both are gone:

- **`server.py` now serves `ui/dist` at `/`** (`_mount_ui`), registered after the API routes with a
  negative-lookahead catch-all `/{tail:(?!api/|ws$).*}` so an unknown `/api/` path still 404s instead
  of quietly answering an HTML page. When `ui/dist` is missing it serves a build hint rather than a
  bare 404. EXECUTED: `/`, `/models`, `/settings` -> 200 text/html; `/api/health`, `/api/config`,
  `/api/summary` -> 200 json; `/api/nope` -> 404.
- **`comfylite.bat`** builds the UI once if needed, then starts the worker. One command, one process,
  one address, no orphan child.
- **`/api/config` now exists.** The Settings panel had been honestly reporting "The worker exposes no
  configuration payload yet" against a route nobody had written. It returns 11 flat rows, and the
  payload is FLAT for a measured reason: `Settings.svelte` drops every entry whose value is an object,
  so a first version that nested the settings under a `config` key rendered exactly one row --
  `source: worker` -- while looking like it had worked.

**Two bugs found by driving it, both mine, both recorded where they bit:**

1. `.page` is a scrolling column flex box and its panels defaulted to `flex-shrink: 1`, so once the
   panels overflowed, every panel was compressed and its text was painted under the next panel's
   header. That is the clipped "Resolved configuration" note in the owner's screenshot. Fixed in
   `Settings.svelte` and `Overview.svelte`; verified in the browser -- `flexShrink: 0`, 0 squashed,
   0 overlaps across all 4 panels.
2. The launcher was first named `run.bat`, and this machine has a `C:\Windows\System32\run.bat`
   (62 bytes, 2026-05-20: `call venv\Scripts\activate.bat` / `python main.py`). System32 is on PATH,
   so it ran instead and failed in a way that reads exactly like a bug in ours. It cost two rounds of
   debugging before `where.exe run.bat` named the real file. **That stray file is still there** --
   removing it is the owner's call, and anything on this bench that ships a `run.bat` should assume
   it will be shadowed.

Still open on this ticket: **the Tauri window**, unchanged. `cargo tauri dev` is a full link against
`~/.cargo` on C: with 55.4 GiB free.
