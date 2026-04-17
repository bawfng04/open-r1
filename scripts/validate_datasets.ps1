param(
    [string]$Manifest = "datasets/datasets.manifest.yaml",
    [string]$Lock = "datasets/datasets.lock.json",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$strictFlag = ""
if ($Strict) {
    $strictFlag = "--strict"
}

python scripts/validate_datasets.py --manifest $Manifest --lock $Lock $strictFlag
