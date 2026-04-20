Param(
  [string]$OutputRoot = "data/datasets/requested",
  [string]$ManifestPath = "datasets/requested_datasets.manifest.yaml",
  [string]$LockPath = "datasets/requested_datasets.lock.json"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

Set-Location $repoRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

python scripts/prepare_requested_datasets.py --output-root $OutputRoot --manifest-path $ManifestPath --lock-path $LockPath
