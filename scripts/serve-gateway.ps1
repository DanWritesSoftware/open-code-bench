<#
.SYNOPSIS
  Start the open-code-bench LiteLLM gateway (Phase 0, pip-based).

.DESCRIPTION
  Launches the LiteLLM proxy from the project's .venv against litellm/config.yaml.

  Sets PYTHONUTF8=1 first: on Windows the console default codepage (cp1252) cannot
  encode LiteLLM's Unicode startup banner, which otherwise crashes startup with a
  UnicodeEncodeError. UTF-8 mode fixes it. (Not needed once the gateway runs in Docker.)

.EXAMPLE
  .\scripts\serve-gateway.ps1
  .\scripts\serve-gateway.ps1 -Port 4001
#>
[CmdletBinding()]
param(
    [string]$BindHost = '127.0.0.1',   # NB: not -Host ($Host is a reserved PowerShell variable)
    [int]$Port        = 4000,
    [string]$Config                      # defaults to <repo>/litellm/config.yaml
)

$ErrorActionPreference = 'Stop'

# Resolve repo root as the parent of this script's directory, so it works from anywhere.
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $RepoRoot 'litellm\config.yaml' }
$LiteLLM = Join-Path $RepoRoot '.venv\Scripts\litellm.exe'

if (-not (Test-Path $LiteLLM)) {
    throw "litellm not found at '$LiteLLM'. Create the venv and run: .venv\Scripts\python.exe -m pip install 'litellm[proxy]'"
}
if (-not (Test-Path $Config)) {
    throw "config not found at '$Config'."
}

# Required on Windows: see .DESCRIPTION above.
$env:PYTHONUTF8 = '1'

# Load .env (gitignored) so config refs like `os.environ/OLLAMA_PI_BASE` resolve without
# hardcoding host IPs in the repo. Copy .env.example -> .env and fill in your values.
$EnvFile = Join-Path $RepoRoot '.env'
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.+?)\s*$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] }
    }
}

Write-Host "Starting LiteLLM gateway on http://${BindHost}:${Port}"
Write-Host "  config: $Config"
& $LiteLLM --config $Config --host $BindHost --port $Port
