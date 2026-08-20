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

while ($true) {
    $lines = @(
        "dono=$Owner"
        "pid=$PID"
        "desde=$since"
        "hb=$([DateTimeOffset]::Now.ToUnixTimeSeconds())"
        "owner_kind=controller"
    )
    Set-Content -Path $LockPath -Value $lines -Encoding ascii -ErrorAction SilentlyContinue
    Start-Sleep -Seconds $IntervalSec
}
