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

# Reads BOTH dialects. `key=value` is canonical and is what everything writes today.
# JSON is what tools/gpu_lock.py used to write, and parsing it here is not politeness --
# it is the direction of the 2026-08-21 defect, MEASURED:
#
#     Owner    :
#     Pid      :
#     StaleFor : 9223372036854775807
#     lock is stale (pid  dead, hb 9223372036854775807s ago) -- reclaiming from
#     lock TAKEN by other-session:steal-test
#
# Every `^$k=(.*)$` missed against a JSON body, so Owner and Pid came back empty and StaleFor
# saturated; Take-GpuLock then read `'' -and (...)` as false and `false -or (MaxValue -lt 55)`
# as false, skipped the held branch, and reclaimed a LIVE lock -- announcing a blank owner,
# which reads exactly like a leftover file from a crashed run.
#
# gpu_lock.py now writes key=value, so this branch should never fire again. It stays because a
# session running an out-of-date copy of that file is the one case where it must.
function Get-GpuLockState {
    if (-not (Test-Path $script:LockPath)) { return $null }
    $raw = Get-Content $script:LockPath -Raw -EA SilentlyContinue
    if ($null -eq $raw) { return $null }

    $owner = ''; $lockPid = ''; $hb = ''; $kind = ''
    if ($raw.TrimStart().StartsWith('{')) {
        try {
            $j = $raw | ConvertFrom-Json
            $owner = [string]$j.owner
            $lockPid = [string]$j.pid
            $kind = 'python-legacy-json'
        } catch {
            # Unparseable body. Report it as held with an unknown owner rather than as absent --
            # a lock we cannot read is not a lock we may take.
            $owner = '(unreadable lock file)'
            $kind = 'unknown'
        }
    } else {
        $c = $raw -split "`r?`n"
        $get = {
            param($k)
            $m = $c | Select-String "^$k=(.*)$"
            if ($m) { $m.Matches.Groups[1].Value.Trim() } else { '' }
        }
        $owner = & $get 'dono'
        $lockPid = & $get 'pid'
        $hb = & $get 'hb'
        $kind = & $get 'owner_kind'
    }

    [pscustomobject]@{
        Owner     = $owner
        Pid       = $lockPid
        Kind      = $kind
        Unreadable = ($kind -eq 'unknown')
        # Distinguir "carimbado ha muito tempo" de "nunca carimbado" e o ponto de HbLido.
        # OBSERVADO 2026-09-01: um lock deixado por uma sessao irma morta (`claude-glm-w4a16-quant`,
        # pid 62788) trouxe `dono=` e `pid=` legiveis e `hb=` ausente, e a mensagem saiu como
        #     lock is stale (pid 62788 dead, hb 9223372036854775807s ago)
        # Aquele numero e `[int64]::MaxValue` e e EXATAMENTE a assinatura do roubo de 2026-08-21,
        # que este arquivo documenta em detalhe -- so que ali Owner e Pid tambem vinham vazios, e
        # aqui nao. O reclaim estava certo (o pid estava morto de verdade e a placa livre), mas
        # quem ler a mensagem vai caçar um bug de parse que nao aconteceu.
        #
        # A garantia de seguranca NAO depende disto e continua valendo: `$alive` sai do `Pid`,
        # que e parseado a parte, entao um irmao VIVO com `hb` ilegivel segue protegido. O que se
        # conserta aqui e so a legibilidade -- que e o que decide se a proxima pessoa acredita na
        # ferramenta.
        HbLido    = [bool]$hb
        StaleFor  = if ($hb) { [DateTimeOffset]::Now.ToUnixTimeSeconds() - [int64]$hb } else { [int64]::MaxValue }
    }
}

function Format-GpuLockHb {
    param($State)
    if ($State.HbLido) { "$($State.StaleFor)s ago" } else { "nunca carimbado" }
}

function Take-GpuLock {
    param([Parameter(Mandatory)][string]$Owner, [int]$StaleLimitSec = 55, [switch]$Force)

    $s = Get-GpuLockState
    if ($s -and -not $Force) {
        $alive = $s.Pid -and (Get-Process -Id $s.Pid -EA SilentlyContinue)
        if ($alive -or $s.StaleFor -lt $StaleLimitSec) {
            Write-Host "lock HELD by $($s.Owner) (pid $($s.Pid) alive=$([bool]$alive), hb $(Format-GpuLockHb $s), kind $($s.Kind)) -- not taking"
            return $false
        }
        # A lock we cannot read, or one that names nobody, is NOT a lock we may reclaim.
        # The 2026-08-21 theft announced exactly this -- "reclaiming from " with a blank owner --
        # and a blank owner is the signature of a parse failure, not of a crashed run. A real
        # leftover lock still names who wrote it.
        if ($s.Unreadable -or -not $s.Owner) {
            Write-Host "lock file present but UNIDENTIFIABLE (owner '$($s.Owner)', kind $($s.Kind)) -- refusing to reclaim."
            Write-Host "  A blank owner means this file was not parsed, not that the run died."
            Write-Host "  Inspect it: Get-Content $script:LockPath"
            return $false
        }
        Write-Host "lock is stale (pid $($s.Pid) dead, hb $(Format-GpuLockHb $s)) -- reclaiming from $($s.Owner)"
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

    # POLL, do not sleep a fixed interval. This was `Start-Sleep -Seconds 2`, which is a race
    # against pwsh startup plus a file write on whatever the machine is doing at that moment --
    # and it lost once here on 2026-08-22, on a box also running eleven ERP containers. Measured
    # afterwards on an idle box the beat was up in under 0.4 s, which is exactly why a fixed
    # sleep is the wrong shape: it passes every time you test it.
    $deadline = [DateTime]::Now.AddSeconds(20)
    $now = $null
    while ([DateTime]::Now -lt $deadline) {
        Start-Sleep -Milliseconds 200
        $now = Get-GpuLockState
        if ($now -and $now.Owner -eq $Owner) { break }
        if (-not (Get-Process -Id $proc.Id -EA SilentlyContinue)) { break }   # it died; stop waiting
    }

    if (-not $now -or $now.Owner -ne $Owner) {
        Stop-Process -Id $proc.Id -Force -EA SilentlyContinue
        # And CLEAN UP. This used to return $false leaving the lock file on disk -- so a failed
        # take locked the card for everyone, naming a pid it had just killed, and the next
        # Assert-GpuLock threw citing a lock that belonged to nobody. That happened here on
        # 2026-08-22 and cost the session its GPU window until the file was removed by hand.
        # Remove it only if it names the beat we just killed: a lock somebody else legitimately
        # holds is never ours to delete, which is the one thing a lock must not get wrong.
        $stray = Get-GpuLockState
        if ($stray -and $stray.Pid -eq $proc.Id) {
            Remove-Item $script:LockPath -Force -EA SilentlyContinue
            Write-Host "heartbeat did not come up -- lock NOT taken (its half-written lock removed)"
        } else {
            Write-Host "heartbeat did not come up -- lock NOT taken"
        }
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
        throw "GPU lock held by $($s.Owner) (pid $($s.Pid), hb $(Format-GpuLockHb $s)) -- refusing to run GPU work."
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
