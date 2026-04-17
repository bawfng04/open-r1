param(
    [string]$Datasets = "math_500,gsm8k,aime24,amc23",
    [int]$NumQuestions = 16,
    [int]$NumGenerations = 8,
    [int]$Seed = 42,
    [string]$RuntimeProfile = "local-dryrun",
    [string]$Method = "vanilla",
    [string]$OutputDir = "data/benchmark-dryrun",
    [switch]$OfflineOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$offlineFlag = ""
if ($OfflineOnly) {
    $offlineFlag = "--offline-only"
}

python scripts/run_benchmark_dryrun.py --datasets $Datasets --num-questions $NumQuestions --num-generations $NumGenerations --seed $Seed --runtime-profile $RuntimeProfile --method $Method --output-dir $OutputDir $offlineFlag
