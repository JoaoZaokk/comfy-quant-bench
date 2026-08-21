"""A window for `cortiq ltx-video`: prompt in, frames on screen.

The engine has no UI. `cortiq serve` has a web dashboard but it serves chat, not video, and the
video path (`ltx-video`, `ltx-render`, `imagine`) is command line only. This is a local page that
drives the same binary and shows what comes back.

    .\\python_embeded\\python.exe -s .\\tools\\ltx_studio.py
    http://127.0.0.1:8123

Binds loopback only. No dependency beyond the embedded interpreter's stdlib plus Pillow, which is
already installed -- tkinter is NOT in this distribution (no tcl/tk), which is why this is a page
and not a window.

WHAT THE PROGRESS BAR CAN AND CANNOT KNOW. cortiq prints `step N/M` per denoise step, so the
denoise phase has real progress. The VAE decode prints one line when it is DONE and nothing while
it runs -- and on a 512x512x49 render that silence is ~259 s of a ~510 s job, half the wall clock.
This shows elapsed time there and says it is opaque rather than drawing a bar that is guessing.
Every phase boundary below is a line the binary actually prints; none of them is a timer.
"""
from __future__ import annotations

import http.server
import json
import re
import shlex
import socketserver
import subprocess
import threading
import time
from pathlib import Path

CORTIQ = Path(r"F:\cortiq-cmf\target\release\cortiq.exe")
MODEL = Path(r"F:\cortiq\ltx25-q4tp.cmf")
OUTROOT = Path(r"F:\cortiq\studio")
PORT = 8123
# `cortiq ltx-video` writes PPM stills and, with --out, a YUV4MPEG2 stream. Neither plays in a
# browser and neither is a file anyone wants to receive after waiting thirteen minutes, so the
# stills get muxed into one h264 mp4 as the last step of a render.
FFMPEG = Path(r"C:\ffmpeg\bin\ffmpeg.exe")

# What cortiq prints, and what each line means. Order matters: `step` before `denoised`.
LINE_STEP = re.compile(r"step\s+(\d+)/(\d+)")
LINE_CONTEXT = re.compile(r"prompt:\s+(\d+) tokens .* in ([\d.]+)s")
LINE_DENOISED = re.compile(r"denoised in ([\d.]+)s")
LINE_DECODED = re.compile(r"decoded (\d+) frames of (\d+)x(\d+) in ([\d.]+)s")
LINE_TOTAL = re.compile(r"^total ([\d.]+)s")


class Job:
    """One render. Owns the child process and the state the page polls."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.phase = "idle"
        self.step = 0
        self.steps = 0
        self.started = 0.0
        self.phase_started = 0.0
        self.lines: list[str] = []
        self.frames: list[str] = []
        self.outdir: Path | None = None
        self.fps = 25
        self.video: str | None = None
        self.audio: str | None = None
        self.error: str | None = None
        self.cmd = ""

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            return {
                "running": self.running,
                "phase": self.phase,
                "step": self.step,
                "steps": self.steps,
                "elapsed": round(now - self.started, 1) if self.started else 0.0,
                "phase_elapsed": round(now - self.phase_started, 1) if self.phase_started else 0.0,
                # The denoise phase is the only one with a real fraction. Everything else
                # reports elapsed time and says so; see the module docstring.
                "fraction": (self.step / self.steps) if self.steps else None,
                "lines": self.lines[-40:],
                "frames": self.frames,
                "video": self.video,
                "error": self.error,
                "cmd": self.cmd,
            }

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = time.time()

    def start(self, opts: dict) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("a render is already running")
            self.reset()
            self.running = True
            self.started = time.time()
            self._set_phase("starting")
            stamp = time.strftime("%H%M%S")
            self.outdir = OUTROOT / f"run_{stamp}"
            self.fps = opts["fps"]
            self.outdir.mkdir(parents=True, exist_ok=True)

        argv = [
            str(CORTIQ), "ltx-video",
            "--model", str(MODEL),
            "--prompt", opts["prompt"],
            "--height", str(opts["height"]),
            "--width", str(opts["width"]),
            "--frames", str(opts["frames"]),
            "--fps", str(opts["fps"]),
            "--seed", str(opts["seed"]),
            "--steps", str(opts["steps"]),
            "--out-dir", str(self.outdir),
        ]
        if opts.get("audio"):
            argv += ["--out-audio", str(self.outdir / "audio.wav")]
        if opts.get("two_stage"):
            argv.append("--two-stage")
        env = {
            "CMF_GPU": "wgpu",
            # Pin the card. Without this wgpu picks its own "best" adapter and on this host that
            # is the 3080 Ti, not the 3090 -- nine measurement runs went to the wrong card that
            # way on 2026-08-21. `cortiq gpu` prints the indices.
            "CMF_GPU_ADAPTER": str(opts["adapter"]),
            # The cooperative-matrix kernel accumulates f32 GEMMs at tf32-class precision:
            # ~14% off the wall clock of a render, measured, at 4.9e-6 relative in the decoded
            # pixels. Off means the device arm reproduces the host arm to nine digits.
            "CMF_COOP": "1" if opts["coop"] else "0",
            # 0 puts every blocked GEMM back in one probe class -- the behaviour before the
            # narrow/wide split. 256 is the split.
            "CMF_GEMM_NT_M": "256" if opts["split"] else "0",
        }
        with self.lock:
            self.cmd = " ".join(f"{k}={v}" for k, v in env.items()) + "  " + shlex.join(argv)
        threading.Thread(target=self._run, args=(argv, env), daemon=True).start()

    def _run(self, argv: list[str], env: dict) -> None:
        import os

        full_env = {**os.environ, **env}
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=full_env,
            )
        except OSError as exc:
            with self.lock:
                self.error = f"could not start {CORTIQ}: {exc}"
                self.running = False
                self._set_phase("failed")
            return

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line or " INFO " in line:
                continue
            with self.lock:
                self.lines.append(line)
                if LINE_CONTEXT.search(line):
                    self._set_phase("denoising")
                elif m := LINE_STEP.search(line):
                    self.step, self.steps = int(m.group(1)), int(m.group(2))
                    self._set_phase("denoising")
                elif LINE_DENOISED.search(line):
                    self._set_phase("decoding")
                elif LINE_DECODED.search(line):
                    self._set_phase("writing frames")
                elif LINE_TOTAL.search(line):
                    self._set_phase("done")

        rc = proc.wait()
        with self.lock:
            frames = sorted(p.name for p in self.outdir.glob("*.ppm")) if self.outdir else []
            self.frames = frames
            if rc != 0:
                self.running = False
                self.error = f"cortiq exited {rc}"
                self._set_phase("failed")
                return
            self._set_phase("encoding mp4")

        # Encoding is seconds against a render measured in minutes, so it runs inline rather
        # than on demand -- and a failure here is reported, not swallowed: the frames are still
        # on disk and saying "no video" beats leaving a broken <video> tag on the page.
        if frames:
            try:
                self._encode(len(frames))
            except Exception as exc:  # ffmpeg missing, or a codec this build lacks
                with self.lock:
                    self.error = f"frames are in {self.outdir}, but the mp4 failed: {exc}"

        with self.lock:
            self.running = False
            self._set_phase("done")

    def _encode(self, count: int) -> None:
        assert self.outdir is not None
        if not FFMPEG.exists():
            raise RuntimeError(f"{FFMPEG} not found")
        mp4 = self.outdir / "render.mp4"
        wav = self.outdir / "audio.wav"
        argv = [
            str(FFMPEG), "-y", "-loglevel", "error",
            "-framerate", str(self.fps),
            "-i", str(self.outdir / "frame_%04d.ppm"),
        ]
        # yuv420p and even dimensions, or the file plays in ffplay and nowhere else.
        if wav.is_file():
            argv += ["-i", str(wav), "-c:a", "aac", "-b:a", "192k", "-shortest"]
        argv += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-movflags", "+faststart", str(mp4)]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError((done.stderr or "").strip()[:300] or f"ffmpeg exited {done.returncode}")
        with self.lock:
            self.video = mp4.name
            self.audio = wav.name if wav.is_file() else None


JOB = Job()


def ppm_to_png(path: Path) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    with Image.open(path) as im:
        im.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


PAGE = """<!doctype html><meta charset=utf-8><title>LTX Studio</title>
<style>
 :root { color-scheme: dark; }
 body { background:#111; color:#ddd; font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
        margin:0; padding:24px; max-width:1100px; }
 h1 { font-size:18px; font-weight:600; margin:0 0 4px; }
 .sub { color:#888; margin-bottom:20px; }
 label { display:block; color:#999; font-size:12px; margin:12px 0 4px; }
 input,textarea,select { background:#1c1c1c; color:#eee; border:1px solid #333; border-radius:6px;
        padding:8px; font:inherit; width:100%; box-sizing:border-box; }
 textarea { min-height:70px; resize:vertical; }
 .row { display:flex; gap:12px; flex-wrap:wrap; }
 .row > div { flex:1 1 110px; }
 button { background:#2d5fd0; color:#fff; border:0; border-radius:6px; padding:10px 20px;
        font:inherit; font-weight:600; cursor:pointer; margin-top:18px; }
 button:disabled { background:#333; color:#777; cursor:default; }
 details { margin-top:16px; border-top:1px solid #262626; padding-top:12px; }
 summary { cursor:pointer; color:#888; font-size:13px; }
 .note { color:#7a7a7a; font-size:12px; margin-top:4px; }
 #bar { height:6px; background:#222; border-radius:3px; overflow:hidden; margin:10px 0 6px; }
 #fill { height:100%; width:0; background:#2d5fd0; transition:width .3s; }
 #phase { font-weight:600; }
 pre { background:#0b0b0b; border:1px solid #222; border-radius:6px; padding:10px;
       max-height:220px; overflow:auto; font-size:12px; color:#9a9a9a; }
 .frames { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
       gap:8px; margin-top:14px; }
 .frames img { width:100%; border-radius:4px; display:block; }
 #player video { width:100%; max-width:640px; border-radius:8px; margin-top:16px; display:block; }
 #player a { color:#6a9bf0; font-size:12px; }
 .err { color:#e06c6c; }
</style>
<h1>LTX Studio</h1>
<div class=sub>Prompt in, video out. Drives <code>cortiq ltx-video</code> directly &mdash; no ComfyUI.</div>

<label>Prompt</label>
<textarea id=prompt>a brass trumpet on a wooden table, morning light, shallow depth of field</textarea>

<div class=row>
  <div><label>Width</label><input id=width type=number value=512 step=64></div>
  <div><label>Height</label><input id=height type=number value=512 step=64></div>
  <div><label>Frames</label><input id=frames type=number value=49></div>
  <div><label>FPS</label><input id=fps type=number value=25></div>
  <div><label>Seed</label><input id=seed type=number value=1234></div>
  <div><label>Steps</label><input id=steps type=number value=8></div>
</div>
<div class=note>
  Steps 8 is the distilled ladder the model was trained on and what a full render costs
  (~13 min at 512&times;512&times;49 here). 3 gives a preview in about a third of that.
</div>
<div class=row>
  <div><label><input type=checkbox id=audio style="width:auto"> Soundtrack</label>
    <div class=note>The model writes a 48 kHz stereo WAV; it gets muxed into the mp4.</div></div>
  <div><label><input type=checkbox id=two_stage style="width:auto"> Two-stage</label>
    <div class=note>Half resolution, latent upscale, then refinement &mdash; how the distilled
    model was trained to sample.</div></div>
</div>

<details>
  <summary>Advanced &mdash; the two switches measured on this bench</summary>
  <div class=row>
    <div><label>Cooperative kernel (tf32)</label>
      <select id=coop><option value=1 selected>on &mdash; ~14% faster</option>
      <option value=0>off &mdash; matches host exactly</option></select></div>
    <div><label>GEMM probe split</label>
      <select id=split><option value=1 selected>on (m&ge;256)</option>
      <option value=0>off &mdash; old behaviour</option></select></div>
    <div><label>GPU adapter</label><input id=adapter type=number value=1></div>
  </div>
  <div class=note>
    tf32 on: ~14% off the wall clock, and the decoded pixels differ from the exact arm by
    4.9e-6 relative (1.2e-4 on the largest single value) &mdash; invisible, not identical.
    Adapter 1 is the RTX 3090 here; 0 is the 3080 Ti. Run <code>cortiq gpu</code> to confirm.
  </div>
</details>

<button id=go>Render</button>

<div id=status style="margin-top:20px">
  <div id=bar><div id=fill></div></div>
  <div><span id=phase>idle</span> <span id=timing class=note></span></div>
  <div id=error class=err></div>
</div>

<pre id=log></pre>
<div id=player></div>
<div class=frames id=frames></div>

<script>
const $ = id => document.getElementById(id);
let polling = null;

$('go').onclick = async () => {
  $('go').disabled = true;
  $('frames').innerHTML = ''; $('player').innerHTML = ''; $('error').textContent = '';
  const body = {
    prompt: $('prompt').value, width: +$('width').value, height: +$('height').value,
    frames: +$('frames').value, fps: +$('fps').value, seed: +$('seed').value,
    steps: +$('steps').value, audio: $('audio').checked, two_stage: $('two_stage').checked,
    coop: $('coop').value === '1', split: $('split').value === '1', adapter: +$('adapter').value,
  };
  const r = await fetch('/render', {method:'POST', body: JSON.stringify(body)});
  if (!r.ok) { $('error').textContent = await r.text(); $('go').disabled = false; return; }
  polling = setInterval(poll, 700); poll();
};

async function poll() {
  const s = await (await fetch('/progress')).json();
  $('phase').textContent = s.phase;
  // A bar only where the binary reports a fraction. The decode prints nothing until it is
  // finished, so it gets elapsed time and an honest label instead of a moving bar.
  if (s.fraction !== null && s.phase === 'denoising') {
    $('fill').style.width = (s.fraction * 100) + '%';
    $('timing').textContent = `step ${s.step}/${s.steps} — ${s.elapsed}s total`;
  } else if (s.phase === 'decoding') {
    $('fill').style.width = '100%';
    $('timing').textContent =
      `${s.phase_elapsed}s in this phase — the VAE reports nothing until it finishes, ` +
      `so there is no fraction to show (${s.elapsed}s total)`;
  } else {
    $('timing').textContent = s.elapsed ? `${s.elapsed}s` : '';
  }
  $('log').textContent = s.lines.join('\\n');
  $('log').scrollTop = $('log').scrollHeight;
  if (s.error) $('error').textContent = s.error;
  if (!s.running) {
    clearInterval(polling); $('go').disabled = false;
    if (s.video) {
      $('player').innerHTML =
        `<video controls autoplay loop src="/video/${encodeURIComponent(s.video)}"></video>` +
        `<a href="/video/${encodeURIComponent(s.video)}" download>download mp4</a>`;
    }
    $('frames').innerHTML = s.frames
      .map(f => `<img loading=lazy src="/frame/${encodeURIComponent(f)}">`).join('');
  }
}
</script>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # the page polls twice a second; do not narrate it
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/progress":
            self._send(200, json.dumps(JOB.snapshot()).encode(), "application/json")
        elif self.path.startswith("/video/"):
            name = self.path[len("/video/"):]
            path = (JOB.outdir / name) if JOB.outdir else None
            if path and path.is_file() and path.suffix == ".mp4":
                self._send(200, path.read_bytes(), "video/mp4")
            else:
                self._send(404, b"no video", "text/plain")
        elif self.path.startswith("/frame/"):
            name = self.path[len("/frame/"):]
            outdir = JOB.snapshot() and JOB.outdir
            path = (outdir / name) if outdir else None
            # Frames come out as PPM. The browser will not render one, so convert on demand
            # rather than writing a second copy of every frame to disk.
            if path and path.is_file() and path.suffix == ".ppm":
                self._send(200, ppm_to_png(path), "image/png")
            else:
                self._send(404, b"no such frame", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/render":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        opts = json.loads(self.rfile.read(length))
        try:
            JOB.start(opts)
        except RuntimeError as exc:
            self._send(409, str(exc).encode(), "text/plain")
            return
        self._send(200, b"{}", "application/json")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    missing = [str(p) for p in (CORTIQ, MODEL) if not p.exists()]
    if missing:
        print("missing, and nothing will render without them:")
        for m in missing:
            print(f"  {m}")
        if not CORTIQ.exists():
            print("\nbuild it:  cargo build --release --features gpu -p cortiq-cli")
        return 1
    OUTROOT.mkdir(parents=True, exist_ok=True)
    print(f"cortiq : {CORTIQ}")
    print(f"model  : {MODEL}")
    print(f"frames : {OUTROOT}")
    print(f"\n  http://127.0.0.1:{PORT}\n")
    if not FFMPEG.exists():
        print(f"note: {FFMPEG} not found -- renders will leave PPM stills and no mp4")
    print("NOT covered: this is a shell over one subcommand. No graph, no LoRA, no nodes --")
    print("             ComfyUI is still the tool for those. Progress is real only during")
    print("             denoise; the VAE decode is silent until it finishes, and on a")
    print("             512x512x49 render that is about half the wall clock.")
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
