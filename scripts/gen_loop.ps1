# Self-resuming generation driver (detached-friendly). Loops `generate --resume` until the run
# dir has all N records. Launch DETACHED via Start-Process so it survives harness task reaping:
#   Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\gen_loop.ps1',
#       '-Spec','specs\livecodebench-32b.yaml','-RunDir','runs\lcb_...','-N','880' -WindowStyle Hidden
param(
    [Parameter(Mandatory)] [string]$Spec,
    [Parameter(Mandatory)] [string]$RunDir,
    [Parameter(Mandatory)] [int]$N
)
$env:PYTHONUTF8 = '1'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo '.venv\Scripts\python.exe'
$rec = Join-Path $RunDir 'records.jsonl'
$log = Join-Path $RunDir '_genloop.log'
for ($i = 1; $i -le 200; $i++) {
    $n = 0
    if (Test-Path $rec) { $n = (Get-Content $rec | Measure-Object -Line).Lines }
    "[loop $i] $n/$N records $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
    if ($n -ge $N) { "[loop] COMPLETE $n/$N" | Out-File -Append -Encoding utf8 $log; break }
    & $py -m ocb.runner.run generate $Spec --resume $RunDir *>> $log
    Start-Sleep -Seconds 2
}
