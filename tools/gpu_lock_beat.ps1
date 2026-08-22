# Heartbeat for F:\GPU_BENCH.lock. Runs detached; its own pid is what the lock
# advertises, which is the whole point -- see gpu_lock.ps1 for why an inline
# `pid=$PID` from a tool call is always a dead pid.
#
# Kept as a FILE rather than a string passed to `pwsh -Command`. The first
# version built the loop with an expandable here-string; `` `n `` inside `@"..."@`
# is consumed by the OUTER string and became a real newline in the generated
# script, which then failed to parse and exited instantly. The lock reported
# "heartbeat did not come up" and nothing was held. A file has no escaping layer.

param(
    [Parameter(Mandatory)][string]$Owner,
    [string]$LockPath = 'F:\GPU_BENCH.lock',
    [int]$IntervalSec = 15
)

$since = [DateTime]::Now.ToString('yyyy-MM-ddTHH:mm:sszzz')
$tmp = "$LockPath.$PID.tmp"
$first = $true

while ($true) {
    if (-not $first) {
        # STOP when the lock stops being ours. The first version of this loop rewrote the file
        # unconditionally, forever. An orphaned heartbeat -- a session that died without calling
        # Release-GpuLock -- would then keep stamping its own name over whatever lock somebody
        # legitimately took afterwards, silently converting a released card into a stolen one.
        # Release-GpuLock kills this process before removing the file, so in the normal path the
        # check never fires; it exists for the path where nobody got to call it.
        if (-not (Test-Path $LockPath)) { exit 0 }          # released -- do not resurrect it
        $now = Get-Content $LockPath -Raw -EA SilentlyContinue
        if ($null -eq $now -or $now -notmatch "(?m)^pid=$PID$") { exit 0 }   # somebody else's now
    }
    $first = $false

    $lines = @(
        "dono=$Owner"
        "pid=$PID"
        "desde=$since"
        "hb=$([DateTimeOffset]::Now.ToUnixTimeSeconds())"
        "owner_kind=controller"
    )
    # Temp + move, never a truncating rewrite in place. A reader that catches Set-Content
    # mid-write sees an empty `pid=` and no `hb=` -- which is exactly the state Take-GpuLock
    # reads as stale. Refreshing the lock must not be what makes it look abandoned.
    Set-Content -Path $tmp -Value $lines -Encoding ascii -ErrorAction SilentlyContinue
    Move-Item -Path $tmp -Destination $LockPath -Force -ErrorAction SilentlyContinue

    Start-Sleep -Seconds $IntervalSec
}
