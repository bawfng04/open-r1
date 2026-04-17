param(
    [string]$LocalConfig = "recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml",
    [string]$H100Config = "recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python scripts/validate_server_ready.py --local-config $LocalConfig --h100-config $H100Config
