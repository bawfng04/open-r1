param(
    [string]$AccelerateConfig = "recipes/accelerate_configs/cpu.yaml",
    [string]$RecipeConfig = "recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONUTF8 = "1"

python scripts/validate_server_ready.py --local-config $RecipeConfig --h100-config "recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"

python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('accelerate') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Missing dependency: accelerate. Run ./scripts/setup_local_env.ps1 first."
    exit 1
}

python -m accelerate.commands.launch --config_file $AccelerateConfig src/open_r1/grpo.py --config $RecipeConfig
