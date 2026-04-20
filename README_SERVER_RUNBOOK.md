# Open-R1 Local + H100 Runbook (Requested Datasets)

This runbook covers:

- local validation and smoke tests
- H100 preflight checks
- train and benchmark execution on server
- common failure patterns and fixes

The requested dataset bundle is expected at:

- `data/datasets/requested/train/combined_train.parquet`
- `data/datasets/requested/test/combined_test.parquet`

## 1) Pipelines Covered

1. `vanilla` GRPO train pipeline
2. `mgrpo` GRPO train pipeline
3. `seed` GRPO train pipeline
4. `amsb` GRPO train pipeline
5. benchmark dry-run pipeline on `math500,gsm8k,aime2025,olympiadbench`

## 2) Config Files

Local dry-run:

- `recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_local_dryrun.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_seed_local_dryrun.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_amsb_local_dryrun.yaml`

H100:

- `recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml`
- `recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml`

## 3) Local Validation (Windows)

Run from repo root `compare/open-r1`.

### 3.1 Prepare / refresh requested datasets

```powershell
python scripts/prepare_requested_datasets.py
```

### 3.2 Preflight check (strict dataset)

```powershell
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml --strict-dataset --print-json
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml --strict-dataset --print-json
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml --strict-dataset --print-json
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml --strict-dataset --print-json
```

### 3.3 Run local smoke tests (4 train pipelines)

```powershell
./scripts/run_local_dryrun.ps1 -RecipeConfig recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml
./scripts/run_local_dryrun.ps1 -RecipeConfig recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_local_dryrun.yaml
./scripts/run_local_dryrun.ps1 -RecipeConfig recipes/Qwen2.5-Math-7B/grpo/config_seed_local_dryrun.yaml
./scripts/run_local_dryrun.ps1 -RecipeConfig recipes/Qwen2.5-Math-7B/grpo/config_amsb_local_dryrun.yaml
```

Expected outputs:

- `data/Qwen2.5-Math-7B-Open-R1-GRPO-DryRun/dry_run_summary.json`
- `data/Qwen2.5-Math-7B-Open-R1-MGRPO-DryRun/dry_run_summary.json`
- `data/Qwen2.5-Math-7B-Open-R1-SEED-DryRun/dry_run_summary.json`
- `data/Qwen2.5-Math-7B-Open-R1-AMSB-DryRun/dry_run_summary.json`

Alternative one-command local flow for A-MSB-GRPO:

```powershell
./scripts/run_amsb_local_pipeline.ps1 -OfflineOnly
```

### 3.4 Run benchmark dry-run (4th pipeline)

```powershell
./scripts/run_benchmark_dryrun.ps1 -Datasets "math500,gsm8k,aime2025,olympiadbench" -NumQuestions 16 -NumGenerations 8 -OutputDir data/benchmark-dryrun-requested
```

Expected output:

- `data/benchmark-dryrun-requested/summary.json`

## 4) Move to H100 Server

Copy repo and dataset bundle to server. Keep the same relative paths if possible.

Minimum required data paths on server:

- `data/datasets/requested/train/combined_train.parquet`
- `data/datasets/requested/test/combined_test.parquet`

## 5) H100 Preflight + Train

Run from repo root on Linux server.

### 5.1 Bootstrap environment

```bash
./scripts/bootstrap/install-drivers.sh
./scripts/bootstrap/setup-env.sh --venv-path .venv
source .venv/bin/activate
```

### 5.2 Verify setup (deps + gpu + dataset)

```bash
./scripts/bootstrap/verify-setup.sh --config recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml
./scripts/bootstrap/verify-setup.sh --config recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml
./scripts/bootstrap/verify-setup.sh --config recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml
./scripts/bootstrap/verify-setup.sh --config recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml
```

Optional direct preflight report:

```bash
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml --accelerate-config recipes/accelerate_configs/zero3.yaml --check-deps --check-gpu --strict-dataset --print-json
```

### 5.3 Start training (4 train pipelines)

```bash
./scripts/deploy/train-grpo.sh --config recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml
./scripts/deploy/train-grpo.sh --config recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml
./scripts/deploy/train-grpo.sh --config recipes/Qwen2.5-Math-7B/grpo/config_seed_h100_prod.yaml
./scripts/deploy/train-grpo.sh --config recipes/Qwen2.5-Math-7B/grpo/config_amsb_h100_prod.yaml
```

Each command runs strict preflight before `accelerate launch`.

## 6) Benchmark on Server (4th pipeline)

```bash
python scripts/run_benchmark_dryrun.py --datasets math500,gsm8k,aime2025,olympiadbench --num-questions 64 --num-generations 8 --runtime-profile h100-prod --method vanilla --output-dir data/benchmark-h100-vanilla
python scripts/run_benchmark_dryrun.py --datasets math500,gsm8k,aime2025,olympiadbench --num-questions 64 --num-generations 8 --runtime-profile h100-prod --method mgrpo --output-dir data/benchmark-h100-mgrpo
python scripts/run_benchmark_dryrun.py --datasets math500,gsm8k,aime2025,olympiadbench --num-questions 64 --num-generations 8 --runtime-profile h100-prod --method seed --output-dir data/benchmark-h100-seed
python scripts/run_benchmark_dryrun.py --datasets math500,gsm8k,aime2025,olympiadbench --num-questions 64 --num-generations 8 --runtime-profile h100-prod --method amsb --output-dir data/benchmark-h100-amsb
```

## 7) Common Errors and Fixes

### 7.1 Missing `vllm` or `flash_attn`

Symptom:

- `Missing Python dependencies for H100 profile: ['vllm', 'flash_attn']`

Fix:

```bash
./scripts/bootstrap/setup-env.sh --venv-path .venv
source .venv/bin/activate
```

### 7.2 No CUDA or wrong GPU type

Symptom:

- `torch.cuda.is_available() is False`
- `no H100 found`

Fix:

- Check NVIDIA runtime and driver: `nvidia-smi`
- Ensure you are on correct H100 node
- Re-run: `./scripts/bootstrap/verify-setup.sh --config ...`

### 7.3 Dataset split/path issues

Symptom:

- train/test split missing
- prompt column missing
- local dataset path not found

Fix:

```bash
python scripts/prepare_requested_datasets.py
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_h100_prod.yaml --strict-dataset --print-json
```

If running on server, verify files exist:

```bash
ls -lah data/datasets/requested/train/combined_train.parquet
ls -lah data/datasets/requested/test/combined_test.parquet
```

### 7.4 Parity mismatch between local and H100 configs

Symptom:

- `Profile parity failed ...`

Fix:

- Keep these keys aligned across `*_local_dryrun.yaml` and `*_h100_prod.yaml`:
  - `dataset_name`
  - `dataset_prompt_column`
  - `method`
  - `reward_funcs`
  - `reward_weights`
  - `system_prompt`

Re-check:

```bash
python scripts/validate_server_ready.py --h100-config recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_h100_prod.yaml --strict-dataset --print-json
```

## 8) Quick Go/No-Go Checklist

- [ ] requested datasets exist in `data/datasets/requested/...`
- [ ] `validate_server_ready.py --strict-dataset` passes for all 4 H100 configs
- [ ] local dry-run summaries exist for vanilla/mgrpo/seed/amsb
- [ ] H100 preflight with `--check-deps --check-gpu` passes
- [ ] train commands launch without preflight failure
- [ ] benchmark summaries are generated per method
