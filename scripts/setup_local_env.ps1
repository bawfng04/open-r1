param(
    [switch]$WithDevExtras
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m pip install --upgrade pip
python -m pip install "accelerate==1.4.0" "trl==0.18.0" "transformers==4.52.3" datasets pyyaml latex2sympy2-extended "math-verify==0.5.2" async-lru

if ($WithDevExtras) {
    python -m pip install -e ".[dev]"
}

Write-Host "Local runtime dependencies installed."
