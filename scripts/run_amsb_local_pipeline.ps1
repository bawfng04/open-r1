param(
    [string]$RecipeConfig = "recipes/Qwen2.5-Math-7B/grpo/config_amsb_local_dryrun.yaml",
    [string]$Datasets = "math500,gsm8k,aime2025,olympiadbench",
    [int]$NumQuestions = 16,
    [int]$NumGenerations = 8,
    [int]$Seed = 42,
    [string]$BenchmarkOutputDir = "data/benchmark-dryrun-amsb",
    [switch]$SkipTrain,
    [switch]$SkipBenchmark,
    [switch]$OfflineOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $SkipTrain) {
    ./scripts/run_local_dryrun.ps1 -RecipeConfig $RecipeConfig
}

if (-not $SkipBenchmark) {
    ./scripts/run_benchmark_dryrun.ps1 -Datasets $Datasets -NumQuestions $NumQuestions -NumGenerations $NumGenerations -Seed $Seed -RuntimeProfile "local-dryrun" -Method "amsb" -OutputDir $BenchmarkOutputDir -OfflineOnly:$OfflineOnly
}
