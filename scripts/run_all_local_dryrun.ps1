param(
    [string]$Methods = "vanilla,mgrpo,seed,amsb",
    [string]$Datasets = "math500,gsm8k,aime2025,olympiadbench",
    [int]$NumQuestions = 16,
    [int]$NumGenerations = 8,
    [int]$Seed = 42,
    [switch]$OfflineOnly,
    [switch]$SkipSetupEnv,
    [switch]$SkipDatasetPrepare,
    [switch]$SkipPreflight,
    [switch]$SkipTrain,
    [switch]$SkipBenchmark,
    [switch]$ContinueOnError,
    [string]$LogDir = "logs/pipeline-local-dryrun"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Write-Stamped {
    param([string]$Level, [string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts][$Level] $Message"
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$transcriptPath = Join-Path $LogDir ("local-dryrun-{0}.log" -f $runId)
Start-Transcript -Path $transcriptPath -Force | Out-Null

$stepIndex = 0
$failedSteps = New-Object System.Collections.Generic.List[string]

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    $script:stepIndex += 1
    $start = Get-Date
    Write-Stamped "START" ("Step {0}: {1}" -f $script:stepIndex, $Name)

    try {
        $global:LASTEXITCODE = 0
        & $Action
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "Exit code: $LASTEXITCODE"
        }
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        Write-Stamped "DONE" ("Step {0} finished in {1}s: {2}" -f $script:stepIndex, $elapsed, $Name)
    }
    catch {
        $elapsed = [int]((Get-Date) - $start).TotalSeconds
        $msg = "Step {0} failed after {1}s: {2} :: {3}" -f $script:stepIndex, $elapsed, $Name, $_
        Write-Stamped "FAIL" $msg
        $script:failedSteps.Add($msg) | Out-Null
        if (-not $ContinueOnError) {
            throw
        }
    }
}

function Parse-Methods {
    param([string]$Value)
    $tokens = @()
    foreach ($item in $Value.Split(",")) {
        $name = $item.Trim().ToLowerInvariant()
        if ($name) {
            $tokens += $name
        }
    }
    return $tokens | Select-Object -Unique
}

$methodSpecs = @{
    "vanilla" = @{
        LocalConfig = "recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml"
        H100Config = "recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml"
        DryRunSummary = "data/Qwen2.5-Math-7B-Open-R1-GRPO-DryRun/dry_run_summary.json"
        BenchmarkOutputDir = "data/benchmark-dryrun-vanilla"
    }
    "mgrpo" = @{
        LocalConfig = "recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_local_dryrun.yaml"
        H100Config = "recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml"
        DryRunSummary = "data/Qwen2.5-Math-7B-Open-R1-MGRPO-DryRun/dry_run_summary.json"
        BenchmarkOutputDir = "data/benchmark-dryrun-mgrpo"
    }
    "seed" = @{
        LocalConfig = "recipes/Qwen2.5-Math-7B/grpo/config_seed_local_dryrun.yaml"
        H100Config = "recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml"
        DryRunSummary = "data/Qwen2.5-Math-7B-Open-R1-SEED-DryRun/dry_run_summary.json"
        BenchmarkOutputDir = "data/benchmark-dryrun-seed"
    }
    "amsb" = @{
        LocalConfig = "recipes/Qwen2.5-Math-7B/grpo/config_amsb_local_dryrun.yaml"
        H100Config = "recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml"
        DryRunSummary = "data/Qwen2.5-Math-7B-Open-R1-AMSB-DryRun/dry_run_summary.json"
        BenchmarkOutputDir = "data/benchmark-dryrun-amsb"
    }
}

$selectedMethods = Parse-Methods -Value $Methods
if ($selectedMethods.Count -eq 0) {
    throw "No methods selected. Use -Methods vanilla,mgrpo,seed,amsb"
}
foreach ($method in $selectedMethods) {
    if (-not $methodSpecs.ContainsKey($method)) {
        throw "Unknown method '$method'. Supported: vanilla,mgrpo,seed,amsb"
    }
}

try {
    Write-Stamped "INFO" "Repo root: $repoRoot"
    Write-Stamped "INFO" ("Methods: {0}" -f ($selectedMethods -join ","))
    Write-Stamped "INFO" ("Transcript: {0}" -f $transcriptPath)

    if (-not $SkipSetupEnv) {
        Invoke-Step "Create/activate .venv and install local dependencies" {
            if (-not (Test-Path ".\\.venv\\Scripts\\python.exe")) {
                Write-Stamped "INFO" "Creating .venv"
                python -m venv .venv
            }
            . .\.venv\Scripts\Activate.ps1
            .\scripts\setup_local_env.ps1
        }
    }
    else {
        if (Test-Path ".\\.venv\\Scripts\\Activate.ps1") {
            Invoke-Step "Activate existing .venv" {
                . .\.venv\Scripts\Activate.ps1
            }
        }
    }

    Invoke-Step "Environment diagnostics" {
        python --version
        python -c "import sys,platform; print('python_exe='+sys.executable); print('platform='+platform.platform())"
        python -c "import torch; print('torch='+torch.__version__); print('cuda_available='+str(torch.cuda.is_available())); print('cuda_device_count='+str(torch.cuda.device_count()))"
    }

    if (-not $SkipDatasetPrepare) {
        Invoke-Step "Prepare requested datasets (download + parquet build)" {
            .\scripts\prepare_requested_datasets.ps1
        }
    }

    if (-not $SkipPreflight) {
        foreach ($method in $selectedMethods) {
            $spec = $methodSpecs[$method]
            Invoke-Step ("Preflight parity and dataset check ({0})" -f $method) {
                python scripts/validate_server_ready.py --h100-config $spec.H100Config --strict-dataset --print-json
            }
        }
    }

    if (-not $SkipTrain) {
        foreach ($method in $selectedMethods) {
            $spec = $methodSpecs[$method]
            Invoke-Step ("Dry-run train wiring check ({0})" -f $method) {
                .\scripts\run_local_dryrun.ps1 -RecipeConfig $spec.LocalConfig -H100Config $spec.H100Config
                if (-not (Test-Path $spec.DryRunSummary)) {
                    throw "Expected dry-run summary missing: $($spec.DryRunSummary)"
                }
                Write-Stamped "INFO" ("Dry-run summary ready: {0}" -f $spec.DryRunSummary)
            }
        }
    }

    if (-not $SkipBenchmark) {
        foreach ($method in $selectedMethods) {
            $spec = $methodSpecs[$method]
            Invoke-Step ("Benchmark dry-run ({0})" -f $method) {
                $params = @{
                    Datasets = $Datasets
                    NumQuestions = $NumQuestions
                    NumGenerations = $NumGenerations
                    Seed = $Seed
                    RuntimeProfile = "local-dryrun"
                    Method = $method
                    OutputDir = $spec.BenchmarkOutputDir
                }
                if ($OfflineOnly) {
                    $params["OfflineOnly"] = $true
                }
                .\scripts\run_benchmark_dryrun.ps1 @params
            }
        }
    }

    Invoke-Step "Print benchmark summary table" {
        $rows = @()
        foreach ($method in $selectedMethods) {
            $summaryPath = Join-Path $repoRoot ($methodSpecs[$method].BenchmarkOutputDir + "/summary.json")
            if (-not (Test-Path $summaryPath)) {
                Write-Stamped "WARN" ("Summary missing for {0}: {1}" -f $method, $summaryPath)
                continue
            }
            $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
            foreach ($datasetProp in $summary.datasets.PSObject.Properties) {
                $metrics = $datasetProp.Value.metrics
                $rows += [PSCustomObject]@{
                    method = $method
                    dataset = $datasetProp.Name
                    pass_at_1 = [double]$metrics."pass@1"
                    pass_at_2 = [double]$metrics."pass@2"
                    pass_at_4 = [double]$metrics."pass@4"
                    pass_at_8 = [double]$metrics."pass@8"
                }
            }
        }

        if ($rows.Count -gt 0) {
            $rows | Sort-Object method, dataset | Format-Table -AutoSize
        }
        else {
            Write-Stamped "WARN" "No benchmark summary rows to print"
        }
    }

    if ($failedSteps.Count -gt 0) {
        Write-Stamped "WARN" "Run completed with failures:"
        foreach ($item in $failedSteps) {
            Write-Stamped "WARN" $item
        }
        exit 1
    }

    Write-Stamped "INFO" "Local dry-run full pipeline completed successfully."
    Write-Stamped "INFO" ("Detailed log: {0}" -f $transcriptPath)
}
finally {
    Stop-Transcript | Out-Null
}
