.PHONY: style quality local_env dryrun dryrun_mgrpo dryrun_seed benchmark_dryrun validate_datasets validate_server_ready

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src tests


# dev dependencies
install:
	uv venv openr1 --python 3.11
	. openr1/bin/activate && uv pip install --upgrade pip && \
	uv pip install vllm==0.8.5.post1 && \
	uv pip install setuptools && \
	uv pip install flash-attn --no-build-isolation && \
	GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

local_env:
	python -m pip install --upgrade pip
	python -m pip install accelerate trl transformers datasets pyyaml

style:
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort $(check_dirs) setup.py

quality:
	ruff check --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort --check-only $(check_dirs) setup.py
	flake8 --max-line-length 119 $(check_dirs) setup.py

test:
	pytest -sv --ignore=tests/slow/ tests/

dryrun:
	ACCELERATE_LOG_LEVEL=info python -m accelerate.commands.launch --config_file recipes/accelerate_configs/cpu.yaml src/open_r1/grpo.py --config recipes/Qwen2.5-Math-7B/grpo/config_local_dryrun.yaml

dryrun_mgrpo:
	ACCELERATE_LOG_LEVEL=info python -m accelerate.commands.launch --config_file recipes/accelerate_configs/cpu.yaml src/open_r1/grpo.py --config recipes/Qwen2.5-Math-7B/grpo/config_mgrpo_local_dryrun.yaml

dryrun_seed:
	ACCELERATE_LOG_LEVEL=info python -m accelerate.commands.launch --config_file recipes/accelerate_configs/cpu.yaml src/open_r1/grpo.py --config recipes/Qwen2.5-Math-7B/grpo/config_seed_local_dryrun.yaml

benchmark_dryrun:
	python scripts/run_benchmark_dryrun.py --offline-only

validate_datasets:
	python scripts/validate_datasets.py

validate_server_ready:
	python scripts/validate_server_ready.py

slow_test:
	pytest -sv -vv tests/slow/

# Evaluation

evaluate:
	$(eval PARALLEL_ARGS := $(if $(PARALLEL),$(shell \
		if [ "$(PARALLEL)" = "data" ]; then \
			echo "data_parallel_size=$(NUM_GPUS)"; \
		elif [ "$(PARALLEL)" = "tensor" ]; then \
			echo "tensor_parallel_size=$(NUM_GPUS)"; \
		fi \
	),))
	$(if $(filter tensor,$(PARALLEL)),export VLLM_WORKER_MULTIPROC_METHOD=spawn &&,) \
	MODEL_ARGS="pretrained=$(MODEL),dtype=bfloat16,$(PARALLEL_ARGS),max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" && \
	if [ "$(TASK)" = "lcb" ]; then \
		lighteval vllm $$MODEL_ARGS "extended|lcb:codegeneration|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	else \
		lighteval vllm $$MODEL_ARGS "lighteval|$(TASK)|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	fi
