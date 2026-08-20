# Take and release F:\GPU_BENCH.lock correctly from a Claude Code session.
#
# WHY THIS EXISTS. On 2026-08-19 this project's cortiq session wrote the lock
# inline from a tool call:
#
#     @"
#     dono=cortiq:ltx_wgpu_audit
#     pid=$PID
#     ...
#     "@ | Set-Content F:\GPU_BENCH.lock
#
# Every PowerShell tool call is its OWN short-lived process. `$PID` there names
# that process, and it dies the instant the call returns -- so the lock named a
# dead pid FROM THE MOMENT IT WAS WRITTEN. It was never valid, either time.
# The heartbeat was worse: refreshed by hand, twice, 45 minutes apart, against a
# 55-second staleness limit. The sibling session read it as leaked and was right,
# and meanwhile the card sat idle behind that lock for roughly forty minutes.
#
# THE FIX. The heartbeat is a real detached process that outlives the tool call,
# and the pid recorded is ITS pid. Both dialects of the protocol then agree:
#
#   controller   judged by hb= freshness (55 s limit)
#   older bash   judged by pid liveness
#
# Note the sibling's measurement that MSYS `kill -0` cannot see a native Windows
# process at all (pid 21432 alive in tasklist, kill -0 reporting dead). So hb= is
# the field that actually carries the signal across dialects; keeping pid= real
# is belt and braces, not the mechanism.
#
# THE RULE, which matters more than the code: take the lock when GPU work is
# about to START, release it when that work ENDS. Holding it while idle -- even
# announced, even for a good reason like "keeping the card for a comparison
# later" -- is a false claim on a shared card. And drop the work BEFORE releasing,
# never after: a free lock over a busy GPU is the same lie pointing the other way.
#
#   . F:\COMFY_PORTABLE\tools\gpu_lock.ps1
#   if (Take-GpuLock -Owner 'cortiq:ltx_bench') { <GPU work>; Release-GpuLock }

$script:LockPath = 'F:\GPU_BENCH.lock'
$script:BeatPath = 'F:\COMFY_PORTABLE\tools\.gpu_lock_beat_pid'

function Get-GpuLockState {
    if (-not (Test-Path $script:LockPath)) { return $null }
    $c = Get-Content $script:LockPath -EA SilentlyContinue
    $get = {
        param($k)
        $m = $c | Select-String "^$k=(.*)$"
        if ($m) { $m.Matches.Groups[1].Value.Trim() } else { '' }
    }
    $hb = & $get 'hb'
    [pscustomobject]@{
        Owner    = & $get 'dono'
        Pid      = & $get 'pid'
        StaleFor = if ($hb) { [DateTimeOffset]::Now.ToUnixTimeSeconds() - [int64]$hb } else { [int64]::MaxValue }
    }
}

function Take-GpuLock {
    param([Parameter(Mandatory)][string]$Owner, [int]$StaleLimitSec = 55, [switch]$Force)

    $s = Get-GpuLockState
    if ($s -and -not $Force) {
        $alive = $s.Pid -and (Get-Process -Id $s.Pid -EA SilentlyContinue)
        if ($alive -or $s.StaleFor -lt $StaleLimitSec) {
            Write-Host "lock HELD by $($s.Owner) (pid $($s.Pid) alive=$([bool]$alive), hb $($s.StaleFor)s ago) -- not taking"
            return $false
        }
        Write-Host "lock is stale (pid $($s.Pid) dead, hb $($s.StaleFor)s ago) -- reclaiming from $($s.Owner)"
    }

    # Detached heartbeat: rewrites the file every 15 s stamping its OWN pid, so
    # the entry stays true for exactly as long as the lock is really held.
    #
    # It lives in a FILE, not in a string passed to `-Command`. The first version
    # built the loop with an expandable here-string, where `` `n `` is eaten by the
    # OUTER string and lands in the generated script as a real newline -- which
    # broke the inner double-quoted string, failed to parse, and exited instantly.
    # The only symptom was "heartbeat did not come up". No escaping layer, no bug.
    $proc = Start-Process pwsh -PassThru -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', 'F:\COMFY_PORTABLE\tools\gpu_lock_beat.ps1',
        '-Owner', $Owner, '-LockPath', $script:LockPath)
    Start-Sleep -Seconds 2
    $now = Get-GpuLockState
    if (-not $now -or $now.Owner -ne $Owner) {
        Stop-Process -Id $proc.Id -Force -EA SilentlyContinue
        Write-Host "heartbeat did not come up -- lock NOT taken"
        return $false
    }
    Set-Content $script:BeatPath $proc.Id -Encoding ascii
    Write-Host "lock TAKEN by $Owner (heartbeat pid $($proc.Id))"
    return $true
}

# Take the lock or STOP the script. Use this instead of Take-GpuLock at the top
# of anything that touches the GPU.
#
# WHY: `Take-GpuLock -Owner x | Out-Null` returns $false when the lock is held,
# prints "not taking" -- and then the script happily runs the benchmark anyway,
# because nobody looked at the return value. That happened here on 2026-08-19,
# on a lock held by a live sibling with a 2-second-old heartbeat, hours after
# the same class of mistake had already been called out. A helper that must be
# checked by hand will eventually not be.
function Assert-GpuLock {
    param([Parameter(Mandatory)][string]$Owner, [switch]$Force)
    if (-not (Take-GpuLock -Owner $Owner -Force:$Force)) {
        $s = Get-GpuLockState
        throw "GPU lock held by $($s.Owner) (pid $($s.Pid), hb $($s.StaleFor)s ago) -- refusing to run GPU work."
    }
}

function Release-GpuLock {
    $mine = $null
    if (Test-Path $script:BeatPath) {
        $mine = (Get-Content $script:BeatPath).Trim()
        Stop-Process -Id $mine -Force -EA SilentlyContinue
        Remove-Item $script:BeatPath -Force -EA SilentlyContinue
        Start-Sleep -Milliseconds 300
    }
    # Only remove the file if it is still OURS. Never clobber a lock somebody
    # else legitimately took after ours went stale.
    $s = Get-GpuLockState
    if ($s -and $mine -and $s.Pid -eq $mine) {
        Remove-Item $script:LockPath -Force -EA SilentlyContinue
        Write-Host "lock RELEASED"
    } elseif ($s) {
        Write-Host "lock belongs to $($s.Owner) (pid $($s.Pid)) -- left alone"
    } else {
        Write-Host "no lock present"
    }
}
