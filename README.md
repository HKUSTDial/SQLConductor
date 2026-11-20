# SQLConductor

Repository for the paper "SQLConductor: Search-to-Policy Learning for Step-wise Text-to-SQL Orchestration".

## Repository Layout

```text
.
├── config/                 # Configurations
├── data/                   # Datasets and artifacts
├── script/                 # Launchers and evaluation scripts
├── src/
│   ├── database/           # Shared database utilities
│   ├── eval/               # Result evaluation functions
│   ├── runner/             # Pipeline entry points
│   ├── system/             # Orchestrator, workers, and LLM clients
│   ├── templates/          # Prompt templates
│   ├── preprocess/         # Dataset preprocessing and task creation
│   ├── mcts/               # MCTS workflow exploration
│   ├── trajectories/       # Trajectory selection and reasoning generation
│   └── train/              # Stability-weighted SFT and Curriculum RL
├── README.md               # Setup and guide (this file)
└── requirements.txt        # Main environment dependencies
```

## 1. Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage the Python environments, keeping the
main project and vLLM separate to avoid dependency conflicts. We also recommend running
long-lived model services in [tmux](https://github.com/tmux/tmux/wiki).

In each new terminal or tmux session, use the repository root directory as the working
directory and activate the environment for the task.

### 1.1 Main environment

We use `.venv` as the main environment for preprocessing, trajectory generation, training,
merging, inference, and evaluation.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### 1.2 vLLM environment

We recommend using a separate environment, `vllm_env/.venv`, to serve the LLMs.

```bash
uv venv vllm_env/.venv --python 3.12
source vllm_env/.venv/bin/activate
uv pip install vllm
deactivate
source .venv/bin/activate
```

Please install a vLLM release compatible with your CUDA driver and models.

The GPU assignments below use a single-machine, eight-GPU setup. For a
different layout, adjust `CUDA_VISIBLE_DEVICES`, the training process counts, and
vLLM's `--tensor-parallel-size` together to match your available GPUs and memory.

## 2. Data and Models

### 2.1 Datasets

Datasets are expected under `data/`. For example, BIRD uses the following layout:

```text
data/
├── bird/
│   ├── train/
│   │   ├── train.json
│   │   ├── train_processed.json
│   │   └── train_databases/<db_id>/<db_id>.sqlite
│   └── dev/
│       ├── dev.json
│       └── dev_databases/<db_id>/<db_id>.sqlite
└── ...
```

Other datasets can be placed alongside BIRD under `data/`, with their paths specified in
the corresponding configs.

### 2.2 Models

Models can be downloaded from Hugging Face. For example, suppose we hope to use Qwen3-8B as the policy
backbone, Qwen2.5-Coder-32B-Instruct as the action model, and Qwen3-Embedding-0.6B for some
preprocessing, you can download them with the following commands:

```bash
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir models/Qwen/Qwen3-8B
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
    --local-dir models/Qwen/Qwen2.5-Coder-32B-Instruct
huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
    --local-dir models/Qwen/Qwen3-Embedding-0.6B
```

If the models are stored elsewhere, update the corresponding paths in the configs.

## 3. Train the Orchestration Policy

With the data and models in place, you can train the orchestration policy.

### 3.1 Training data preprocessing and MCTS workflow exploration

First, launch the action-LLM service in tmux. For example, to serve
Qwen2.5-Coder-32B-Instruct:

```bash
tmux new -s sqlconductor-action -c "$PWD"
```

In the session, run:

```bash
source vllm_env/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
    models/Qwen/Qwen2.5-Coder-32B-Instruct \
    --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 9999 \
    --tensor-parallel-size 4 \
    --max-model-len 32000
```

In a separate terminal, we can create a tmux session for the embedding service:

```bash
tmux new -s sqlconductor-embedding -c "$PWD"
```

In the session, run:

```bash
source vllm_env/.venv/bin/activate
CUDA_VISIBLE_DEVICES=4 vllm serve \
    models/Qwen/Qwen3-Embedding-0.6B \
    --served-model-name Qwen/Qwen3-Embedding-0.6B \
    --runner pooling \
    --convert embed \
    --port 9998 \
    --max-model-len 8192
```

Once both services are running, we can preprocess BIRD-Train from the main environment
in another terminal:

```bash
source .venv/bin/activate
python data/dataset_processing/bird_train_preprocessor.py
bash script/preprocess/preprocess_train.sh
```

After preprocessing finishes, stop the embedding service and keep the action-LLM service
running for MCTS and trajectory generation.

Next, add the matching endpoint to `mcts_model_kwargs` in
`config/mcts/qwen2.5_coder_32b_instruct_mcts.yaml`:

```yaml
mcts_model_kwargs:
  model: "Qwen/Qwen2.5-Coder-32B-Instruct"
  base_url: "http://localhost:9999/v1"
  api_key: "dummy"
```

With the service running, we can run MCTS from the main environment to search for workflows and trajectories:

```bash
source .venv/bin/activate
bash script/mcts/qwen2.5_coder_32b_instruct_mcts_bird_train.sh
```

### 3.2 Supervision curation and reasoning generation

With the action-LLM service still running, point `PKL_DIR` in
`script/train/traj/traj_sft_data_processing.sh` to the MCTS result directory.
Then generate the curated trajectories and reasoning data with:

```bash
bash script/train/traj/traj_sft_data_processing.sh
```

### 3.3 Stability-weighted SFT

After trajectory generation, the action-LLM service can be stopped to free GPU resources. 
Before training, update `model.name` in `config/train/sft/sft_qwen3_8b.yaml` to the
local policy-model path and `data.train_path` to the generated trajectory file. Also set
the following in `script/train/sft/meta_agent_sft_qwen3_8b.sh`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Then train on all eight GPUs:

```bash
bash script/train/sft/meta_agent_sft_qwen3_8b.sh
```

After training, we can merge the SFT adapter:

```bash
bash script/train/merge_model.sh \
    --base_model models/Qwen/Qwen3-8B \
    --lora_path checkpoints/sft/sft_qwen3_8b_v1 \
    --output_path checkpoints/merged/sft_qwen3_8b_v1 \
    --disable_nccl_p2p --disable_nccl_ib
```

### 3.4 Curriculum RL

Since RL requires the action-LLM service and the policy trainer to run concurrently,
restart the action service on GPUs 0–3 in tmux:

```bash
source vllm_env/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
    models/Qwen/Qwen2.5-Coder-32B-Instruct \
    --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 9999 \
    --tensor-parallel-size 4 \
    --max-model-len 32000
```

The RL stage is initialized from the merged SFT checkpoint. Review the paths and model
endpoints in `config/train/rl/grpo_qwen3_8b.yaml`, and assign the remaining GPUs to
training in `script/train/rl/meta_agent_rl_qwen3_8b.sh`:

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

RL selects training questions whose MCTS success ratio (correct trajectories / total
trajectories) is strictly below `rl.training.curriculum_threshold`, set to `0.5`
in the provided config. Set `system.paths.oracle_summary` to the complete MCTS
trajectory summary, including questions with no correct trajectories.

Then we launch the RL training with:

```bash
bash script/train/rl/meta_agent_rl_qwen3_8b.sh
```

After RL training finishes, we can merge the best RL adapter:

```bash
bash script/train/merge_model.sh \
    --base_model checkpoints/merged/sft_qwen3_8b_v1 \
    --lora_path checkpoints/rl_gdpo_qwen3_8b_v1/checkpoint-best \
    --output_path checkpoints/merged/sqlconductor-policy \
    --cuda_visible_devices 4 \
    --disable_nccl_p2p --disable_nccl_ib
```

## 4. Preprocessing for Testing and Evaluation

Before evaluation, the evaluation split needs to be preprocessed. For example, on
BIRD-Dev, set the embedding model path and the action-LLM endpoint in
`config/preprocess/bird_dev.toml`, and start the action-LLM service as described above.
Then run the preprocessing pipeline:

```bash
bash script/preprocess/inference_preprocess_bird_dev.sh --all
```

## 5. Inference and Evaluation

With the action-LLM service still running, serve the trained policy in a second tmux
session:

```bash
tmux new -s sqlconductor-policy -c "$PWD"
```

In the session, run:

```bash
source vllm_env/.venv/bin/activate
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve \
    checkpoints/merged/sqlconductor-policy \
    --served-model-name SQLConductor-Policy \
    --port 18001 \
    --tensor-parallel-size 4 \
    --max-model-len 32000
```

Before running inference, update the evaluation task file, database root, and both model
endpoints in the inference config, and use `SQLConductor-Policy` as the policy's served
model name. For example, on BIRD-Dev, use
`config/exp_train/bird_dev_qwen3_8b_trained.yaml`.

We recommend running a smoke test before the full experiment:

```bash
python -m src.runner.sqlconductor_runner \
    --config config/exp_train/bird_dev_qwen3_8b_trained.yaml \
    --max-tasks 8 \
    --num-workers 2 \
    --agent-type meta_agent \
    --output-dir results/inference/bird_dev_smoke
```

After the smoke test, launch the full experiment with:

```bash
python -m src.runner.sqlconductor_runner \
    --config config/exp_train/bird_dev_qwen3_8b_trained.yaml \
    --num-workers 16 \
    --agent-type meta_agent \
    --output-dir results/inference/bird_dev
```

Once the experiment run is complete, you can compute execution accuracy (EX) for the
generated SQL queries, for example:

```bash
bash script/bird_eval.sh results/inference/bird_dev/answer.json
```
