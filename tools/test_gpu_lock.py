"""Executable regression for the GPU lock, both dialects, both directions.

WHY THIS FILE EXISTS. On 2026-08-21 an audit found -- and then a run confirmed -- that
`gpu_lock.py` wrote JSON while `gpu_lock.ps1` parsed `key=value`, so the PowerShell side read a
live Python-held lock as stale and reclaimed it, announcing a blank owner. The theft was one-way,
which is why it survived: `open("x")` meant the Python side always refused correctly, so one
direction worked perfectly and nobody had a reason to test the other.

That is the lesson this file encodes. **A lock has two directions and reading either half in
isolation never reveals a format disagreement -- only running both against each other does.** So
every scenario below drives one implementation against a lock the *other* one wrote.

Nothing here touches the real `F:/GPU_BENCH.lock` or the GPU. The PowerShell scripts are copied to
a temp directory with their hardcoded paths rewritten, so a run of this test cannot disturb a
sibling session that is genuinely holding the card.

    .\\python_embeded\\python.exe -s .\\tools\\test_gpu_lock.py

Exits non-zero on the first failure. No pytest -- it is not installed in the embedded interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gpu_lock  # noqa: E402
from gpu_lock import GpuLock, GpuLockBusy, _parse, _render, read_state, describe  # noqa: E402

REAL_LOCK = "F:" + chr(92) + "GPU_BENCH.lock"
REAL_BEATPID = "F:" + chr(92) + "COMFY_PORTABLE" + chr(92) + "tools" + chr(92) + ".gpu_lock_beat_pid"
REAL_BEAT = "F:" + chr(92) + "COMFY_PORTABLE" + chr(92) + "tools" + chr(92) + "gpu_lock_beat.ps1"

NL = chr(10)

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def rule(t: str) -> None:
    print()
    print("-" * 76)
    print(t)
    print("-" * 76)


class Sandbox:
    """Copies of both .ps1 files pointed at a throwaway lock path."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = root / "GPU_BENCH.lock"
        self.beatpid = root / ".gpu_lock_beat_pid"
        self.beat = root / "gpu_lock_beat.ps1"
        self.ps1 = root / "gpu_lock.ps1"
        shutil.copy(HERE / "gpu_lock_beat.ps1", self.beat)
        text = (HERE / "gpu_lock.ps1").read_text(encoding="utf-8")
        for old, new in ((REAL_LOCK, str(self.lock)),
                         (REAL_BEATPID, str(self.beatpid)),
                         (REAL_BEAT, str(self.beat))):
            assert old in text, f"anchor missing from gpu_lock.ps1: {old!r}"
            text = text.replace(old, new)
        self.ps1.write_text(text, encoding="utf-8")

    def pwsh(self, body: str) -> subprocess.CompletedProcess:
        script = f". '{self.ps1}'\n{body}\n"
        return subprocess.run(
            ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, timeout=120,
        )

    def kill_beat(self) -> None:
        if self.beatpid.is_file():
            pid = self.beatpid.read_text(encoding="ascii").strip()
            subprocess.run(["pwsh", "-NoProfile", "-Command",
                            f"Stop-Process -Id {pid} -Force -EA SilentlyContinue"],
                           capture_output=True, timeout=60)
            self.beatpid.unlink(missing_ok=True)
        self.lock.unlink(missing_ok=True)


HOLDER = r"""
import sys, time
sys.path.insert(0, r"{tools}")
from gpu_lock import GpuLock
from pathlib import Path
with GpuLock("{owner}", path=Path(r"{lock}")):
    print("HELD", flush=True)
    time.sleep({secs})
"""


def spawn_python_holder(sb: Sandbox, owner: str, secs: int) -> subprocess.Popen:
    code = HOLDER.format(tools=str(HERE), owner=owner, lock=str(sb.lock), secs=secs)
    p = subprocess.Popen([sys.executable, "-s", "-c", code],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 20
    while time.time() < deadline:
        if sb.lock.is_file():
            return p
        if p.poll() is not None:
            raise RuntimeError(f"holder died: {p.communicate()[1]}")
        time.sleep(0.1)
    raise RuntimeError("holder never created the lock")


def main() -> int:
    print("=" * 76)
    print("GPU lock regression -- both dialects, both directions")
    print(f"sandbox lock path (never the real one): under {tempfile.gettempdir()}")
    print("=" * 76)

    with tempfile.TemporaryDirectory(prefix="gpulock_test_") as td:
        sb = Sandbox(Path(td))

        # ---------------------------------------------------------- parsing
        rule("1. the parser understands both dialects, and neither silently yields nothing")
        kv = _parse(_render("bench:x", 4242, "2026-08-22T06:00:00-0300"))
        check("key=value: owner", kv.get("dono") == "bench:x", repr(kv.get("dono")))
        check("key=value: pid", kv.get("pid") == "4242", repr(kv.get("pid")))
        check("key=value: hb present", bool(kv.get("hb")), repr(kv.get("hb")))

        legacy = _parse('{\n  "owner": "old:python",\n  "pid": 999999,\n  "started": "x"\n}')
        check("legacy JSON: owner recovered", legacy.get("dono") == "old:python", repr(legacy))
        check("legacy JSON: pid recovered", legacy.get("pid") == "999999", repr(legacy.get("pid")))
        check("legacy JSON: labelled as legacy",
              legacy.get("owner_kind") == "python-legacy-json", repr(legacy.get("owner_kind")))

        # ---------------------------------------------------------- CACHE-02
        rule("2. CACHE-02: a pid must not match another pid by substring")
        sb.lock.write_text(_render("bench:other", 4242, "x"), encoding="ascii")
        state = read_state(sb.lock)
        check("pid 424 does NOT match a lock held by 4242",
              int(state["pid"]) != 424 and state["pid"] == "4242",
              "integer comparison, not `'pid=424' in text`")
        lk = GpuLock("bench:pretender", path=sb.lock)
        lk.held = True          # pretend we hold it, with THIS process's pid
        lk.__exit__()
        check("__exit__ refuses to unlink a lock owned by someone else", sb.lock.is_file(),
              "file survived" if sb.lock.is_file() else "FILE WAS DELETED -- the bug is back")
        sb.lock.unlink(missing_ok=True)

        rule("2b. the two halves must agree about whether a pid is alive")
        # They did not. `_pid_alive` caught ProcessLookupError for dead -- which is the POSIX
        # answer. On Windows a dead pid raises a bare OSError with winerror 87, so it fell to
        # `except OSError: return True` and EVERY dead pid read as alive, while PowerShell's
        # `Get-Process -Id` said dead for the same number. Safe direction, wrong message, and the
        # same split between the two halves that this whole file exists to catch.
        dead = subprocess.Popen([sys.executable, "-s", "-c", "pass"])
        dead.wait(timeout=20)
        check("a pid that has exited reads as dead", not gpu_lock._pid_alive(dead.pid),
              f"pid {dead.pid}")
        check("this process reads as alive", gpu_lock._pid_alive(os.getpid()))
        check("a pid that never existed reads as dead", not gpu_lock._pid_alive(999_999))
        r = sb.pwsh(f"Write-Output \"PS=$([bool](Get-Process -Id {dead.pid} -EA SilentlyContinue))\"")
        ps_says = "PS=True" in r.stdout
        check("and PowerShell agrees with Python about that same pid",
              ps_says == gpu_lock._pid_alive(dead.pid),
              f"PowerShell alive={ps_says}, Python alive={gpu_lock._pid_alive(dead.pid)}")

        # ---------------------------------------------------------- direction A
        rule("3. DIRECTION A -- Python holds, PowerShell asks. This is the one that was broken.")
        holder = spawn_python_holder(sb, "bench:long_sweep", 45)
        try:
            state = read_state(sb.lock)
            print(f"  lock says: {describe(state)}")

            r = sb.pwsh("$s = Get-GpuLockState; "
                        "Write-Output \"OWNER=$($s.Owner)|PID=$($s.Pid)|STALE=$($s.StaleFor)|KIND=$($s.Kind)\"")
            line = next((l for l in r.stdout.splitlines() if l.startswith("OWNER=")), "")
            print(f"  Get-GpuLockState -> {line}")
            check("PowerShell sees the real owner", "OWNER=bench:long_sweep" in line, line)
            check("PowerShell sees the real pid", f"PID={holder.pid}" in line, line)
            check("StaleFor is a real age, not int64::MaxValue",
                  "STALE=9223372036854775807" not in line, line)

            r = sb.pwsh("$got = Take-GpuLock -Owner 'other:steal-attempt'; Write-Output \"TOOK=$got\"")
            print("  " + " / ".join(l for l in r.stdout.splitlines() if l.strip())[:200])
            check("Take-GpuLock REFUSES a live Python lock", "TOOK=False" in r.stdout, r.stdout.strip()[-160:])
            check("it does not print 'reclaiming'", "reclaiming" not in r.stdout.lower())

            r = sb.pwsh("try { Assert-GpuLock -Owner 'other:steal-attempt'; Write-Output 'NOTHROW' } "
                        "catch { Write-Output \"THREW: $($_.Exception.Message)\" }")
            threw = "THREW:" in r.stdout
            check("Assert-GpuLock THROWS", threw,
                  next((l for l in r.stdout.splitlines() if l.startswith("THREW")), r.stdout)[:150])

            still = read_state(sb.lock)
            check("the lock is still the Python holder's after all that",
                  still is not None and still.get("dono") == "bench:long_sweep", describe(still or {}))

            print("  waiting 20 s to watch the heartbeat refresh hb=...")
            hb0 = int(read_state(sb.lock)["hb"])
            time.sleep(20)
            hb1 = int(read_state(sb.lock)["hb"])
            check("heartbeat thread refreshes hb", hb1 > hb0, f"{hb0} -> {hb1} (+{hb1 - hb0}s)")
        finally:
            holder.terminate()
            holder.wait(timeout=20)
        sb.lock.unlink(missing_ok=True)

        # ---------------------------------------------------------- direction B
        rule("4. DIRECTION B -- PowerShell holds, Python asks. This one always worked; prove it still does.")
        r = sb.pwsh("$got = Take-GpuLock -Owner 'controller:bench'; Write-Output \"TOOK=$got\"")
        check("PowerShell can still take a free lock", "TOOK=True" in r.stdout, r.stdout.strip()[-160:])
        if sb.lock.is_file():
            print(f"  lock says: {describe(read_state(sb.lock))}")
            try:
                with GpuLock("bench:intruder", path=sb.lock):
                    check("Python REFUSES a PowerShell-held lock", False, "it took the lock")
            except GpuLockBusy as exc:
                check("Python REFUSES a PowerShell-held lock", True, str(exc).splitlines()[0][:130])
            check("the refusal names the real owner",
                  "controller:bench" in describe(read_state(sb.lock) or {}))

            r = sb.pwsh("Release-GpuLock")
            check("Release-GpuLock releases its own lock",
                  not sb.lock.is_file(), r.stdout.strip()[-120:])
        sb.kill_beat()

        # ---------------------------------------------------------- legacy
        rule("5. a lock in the OLD JSON dialect, with a LIVE pid, must not be reclaimed")
        holder = spawn_python_holder(sb, "unused", 40)
        try:
            sb.lock.write_text(
                '{\n  "owner": "old:python:json",\n  "pid": %d,\n  "started": "x"\n}' % holder.pid,
                encoding="utf-8")
            r = sb.pwsh("$got = Take-GpuLock -Owner 'other:steal'; Write-Output \"TOOK=$got\"")
            print("  " + " / ".join(l for l in r.stdout.splitlines() if l.strip())[:220])
            check("Take-GpuLock refuses a legacy-JSON lock with a live pid",
                  "TOOK=False" in r.stdout, r.stdout.strip()[-160:])
        finally:
            holder.terminate()
            holder.wait(timeout=20)
        sb.kill_beat()

        rule("6. a FAILED take must not leave the card locked -- and must not clean up someone else's")
        # The bug this covers cost a real GPU window on 2026-08-22. Take-GpuLock's heartbeat check
        # failed, it killed the beat, returned $false -- and left the lock file on disk naming the
        # pid it had just killed. The next Assert-GpuLock threw citing a lock that belonged to
        # nobody, and the card stayed locked until the file was removed by hand.
        beat_src = sb.beat.read_text(encoding="utf-8")

        # (a) a beat that comes up and writes, but under a name Take-GpuLock is not expecting.
        #     That is the branch: owner mismatch -> kill -> the stray names our beat -> remove it.
        sb.beat.write_text(
            "param([string]$Owner, [string]$LockPath, [int]$IntervalSec = 15)" + NL
            + "Set-Content -Path $LockPath -Encoding ascii -Value @("
            + "'dono=NOT-THE-OWNER-ASKED-FOR', \"pid=$PID\", 'desde=x',"
            + " \"hb=$([DateTimeOffset]::Now.ToUnixTimeSeconds())\", 'owner_kind=controller')" + NL
            + "while ($true) { Start-Sleep -Seconds 60 }" + NL,
            encoding="utf-8")
        r = sb.pwsh("$got = Take-GpuLock -Owner 'bench:will-not-match'; Write-Output \"TOOK=$got\"")
        check("a take whose heartbeat misbehaves returns false", "TOOK=False" in r.stdout,
              r.stdout.strip()[-140:])
        check("and it does NOT leave the card locked", not sb.lock.is_file(),
              "lock removed" if not sb.lock.is_file() else "LOCK LEFT BEHIND -- the bug is back")
        sb.kill_beat()

        # (b) the same failure, but the lock on disk belongs to somebody else. Never ours to
        #     delete: that is the one failure mode a lock must not have, and the cleanup added
        #     for (a) is exactly the kind of code that gets it wrong.
        sb.beat.write_text(
            "param([string]$Owner, [string]$LockPath, [int]$IntervalSec = 15)" + NL
            + "exit 0" + NL, encoding="utf-8")
        holder = spawn_python_holder(sb, "someone-else:real-work", 30)
        try:
            before = sb.lock.read_text(encoding="utf-8")
            r = sb.pwsh("$got = Take-GpuLock -Owner 'bench:intruder'; Write-Output \"TOOK=$got\"")
            check("a take that fails against a live foreign lock still returns false",
                  "TOOK=False" in r.stdout, r.stdout.strip()[-140:])
            check("and leaves that foreign lock untouched",
                  sb.lock.is_file() and sb.lock.read_text(encoding="utf-8") == before,
                  "intact" if sb.lock.is_file() else "SOMEBODY ELSE'S LOCK WAS DELETED")
        finally:
            holder.terminate()
            holder.wait(timeout=20)
        sb.beat.write_text(beat_src, encoding="utf-8")
        sb.kill_beat()

        rule("7. an UNREADABLE lock file is held, not free")
        sb.lock.write_text("this is not a lock, it is garbage\n", encoding="ascii")
        r = sb.pwsh("$got = Take-GpuLock -Owner 'other:steal'; Write-Output \"TOOK=$got\"")
        print("  " + " / ".join(l for l in r.stdout.splitlines() if l.strip())[:220])
        check("Take-GpuLock refuses a lock it cannot identify", "TOOK=False" in r.stdout,
              r.stdout.strip()[-160:])
        check("and says so instead of claiming the run died",
              "UNIDENTIFIABLE" in r.stdout or "refusing" in r.stdout.lower())

        rule("8. a lock with `dono=` and `pid=` but NO `hb=` reads as 'nunca carimbado'")
        # OBSERVADO 2026-09-01, na maquina real: um lock deixado por uma sessao irma morta
        # (`claude-glm-w4a16-quant`, pid 62788) tinha dono e pid legiveis e `hb=` ausente, e a
        # mensagem saiu como `hb 9223372036854775807s ago`. Aquele numero e [int64]::MaxValue e e
        # a assinatura EXATA do roubo de 2026-08-21 que este arquivo testa acima -- so que la
        # Owner e Pid tambem vinham vazios. Reclamar estava certo (o pid estava morto), mas a
        # mensagem manda a proxima pessoa caçar um bug de parse que nao houve.
        #
        # As duas metades sao testadas separadas de proposito, porque so a segunda e a garantia:
        # a legibilidade e cosmetica, a protecao do irmao VIVO nao e, e ela nao pode depender de
        # `hb` ter sido lido.
        sb.kill_beat()
        sb.lock.write_text("dono=irmao:sem-hb" + NL + "pid=999999" + NL
                           + "desde=x" + NL + "owner_kind=controller" + NL, encoding="ascii")
        r = sb.pwsh("$s = Get-GpuLockState; "
                    "Write-Output \"HB=$(Format-GpuLockHb $s)\"; "
                    "Write-Output \"LIDO=$($s.HbLido)\"; Write-Output \"DONO=$($s.Owner)\"")
        check("Get-GpuLockState nao inventa uma idade para um hb ausente",
              "HB=nunca carimbado" in r.stdout and "LIDO=False" in r.stdout,
              r.stdout.strip()[-140:])
        check("e o dono continua sendo lido (e o que separa isto do defeito de 2026-08-21)",
              "DONO=irmao:sem-hb" in r.stdout, r.stdout.strip()[-140:])
        r = sb.pwsh("$got = Take-GpuLock -Owner 'outro:tentativa'; Write-Output \"TOOK=$got\"")
        check("com pid morto e hb ausente, reclamar e correto e a mensagem diz por que",
              "TOOK=True" in r.stdout and "nunca carimbado" in r.stdout
              and "9223372036854775807" not in r.stdout, r.stdout.strip()[-160:])
        sb.kill_beat()

        # A garantia, testada a parte: pid VIVO e hb ausente tem de segurar. Se isto cair, o
        # `hb` ilegivel virou porta de roubo, que e a coisa que este arquivo inteiro existe para
        # impedir.
        vivo = subprocess.Popen([sys.executable, "-s", "-c",
                                 "import time; time.sleep(30)"])
        try:
            sb.lock.write_text("dono=irmao:vivo-sem-hb" + NL + f"pid={vivo.pid}" + NL
                               + "desde=x" + NL + "owner_kind=controller" + NL, encoding="ascii")
            antes = sb.lock.read_text(encoding="utf-8")
            r = sb.pwsh("$got = Take-GpuLock -Owner 'outro:roubo'; Write-Output \"TOOK=$got\"")
            check("um irmao VIVO com hb ausente NAO e roubado", "TOOK=False" in r.stdout,
                  r.stdout.strip()[-160:])
            check("e o lock dele fica intacto",
                  sb.lock.is_file() and sb.lock.read_text(encoding="utf-8") == antes,
                  "intacto" if sb.lock.is_file() else "O LOCK DO IRMAO FOI APAGADO")
        finally:
            vivo.terminate()
            vivo.wait(timeout=20)
        sb.kill_beat()
        sb.kill_beat()

    print()
    print("=" * 76)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print()
    print("NOT covered by this file: it never touches the real F:/GPU_BENCH.lock, never takes the")
    print("GPU, and does not test the 55-second staleness path (that needs a lock older than the")
    print("limit with a dead pid, which costs a minute of wall clock). It also does not test two")
    print("PowerShell sessions racing each other for a free lock -- Take-GpuLock now POLLS for its")
    print("heartbeat instead of sleeping a fixed 2 s (that sleep lost a race here on 2026-08-22")
    print("and was measured at 0.4 s on an idle box afterwards, which is why a fixed sleep passes")
    print("every time you test it), but the take itself is still not atomic and nothing here")
    print("drives two takers at once.")
    print("=" * 76)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
