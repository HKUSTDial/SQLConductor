from src.mcts.mcts import MCTSSolver
from src.mcts.reward import MajorityVoteRewardModel
from src.runner.task import Task
from src.config.mcts_config import MCTSConfig
from pathlib import Path
from typing import Union
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import yaml
from src.llm_call.openai_llm import DEFAULT_COST_RECORDER
import json
import random
from dotenv import load_dotenv  
import os
import traceback

load_dotenv(override=True)

try:
    import weave
    if "WANDB_API_KEY" in os.environ:
        weave.init("mcts_runner")
    else:
        print("Weave logging disabled: WANDB_API_KEY is unset.")
except ImportError:
    print("Weave logging disabled: package is unavailable.")

class MCTSRunner:
    def __init__(self, config: Union[MCTSConfig, str]):
        if isinstance(config, str):
            config_path = Path(config)
            assert config_path.exists(), f"Config file {config_path} does not exist"
            if config_path.suffix == ".json":
                self.config = MCTSConfig.model_validate_json(config_path.read_text())
            elif config_path.suffix == ".yaml":
                self.config = MCTSConfig.model_validate(yaml.safe_load(config_path.read_text()))
            else:
                raise ValueError(f"Unsupported config file extension: {config_path.suffix}")
        else:
            self.config = config
            
        if not Path(self.config.save_root_dir).exists():
            Path(self.config.save_root_dir).mkdir(parents=True, exist_ok=True)

        random.seed(self.config.random_seed)
        
    def run_one_task(self, task: Task) -> str:
        mcts_solver = MCTSSolver(
            db_root_dir=self.config.db_root_dir,
            task=task,
            max_rollout_steps=self.config.max_rollout_steps,
            max_depth=self.config.max_depth,
            exploration_constant=self.config.exploration_constant,
            save_root_dir=self.config.save_root_dir,
            llm_kwargs=self.config.mcts_model_kwargs,
            reward_model=MajorityVoteRewardModel(self.config.reward_model_kwargs)
        )
        try:
            mcts_solver.solve()
        except Exception as e:
            print(f"Error solving task {task.question_id}: {e}")
            traceback.print_exc()
        DEFAULT_COST_RECORDER.print_profile()
    
    def run_all_tasks(self):
        with open(self.config.tasks_file_path, "rb") as f:
            tasks = pickle.load(f)
            
        if self.config.subset_file_path:
            print(f"Using subset file {self.config.subset_file_path} to filter tasks")
            with open(self.config.subset_file_path, "r") as f:
                subset_data = json.load(f)
                subset_ids = [item["question_id"] for item in subset_data]
                tasks = [task for task in tasks if task.question_id in subset_ids]
            print(f"Tasks after filtering: {len(tasks)}")
        
        done_task_ids = []
        for pkl_file in Path(self.config.save_root_dir).glob("*.pkl"):
            done_task_ids.append(int(pkl_file.stem))
        print(f"Skipping {len(done_task_ids)} completed tasks")
        tasks = [task for task in tasks if task.question_id not in done_task_ids]
        
        with open(Path(self.config.save_root_dir) / "config.json", "w") as f:
            print(f"Saving config to {Path(self.config.save_root_dir) / 'config.json'}")
            json.dump(self.config.model_dump(), f, indent=4)

        print(f"There are {len(tasks)} tasks to solve")
        with ProcessPoolExecutor(max_workers=self.config.n_processes) as executor:
            list(tqdm(executor.map(self.run_one_task, tasks), total=len(tasks), desc="Solving tasks"))

        print("All tasks finished. Calculating total tokens and time...")
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_time = 0
        for info_file in Path(self.config.save_root_dir).glob("*_info.json"):
            try:
                with open(info_file, "r") as f:
                    info = json.load(f)
                    total_prompt_tokens += info.get("prompt_tokens", 0)
                    total_completion_tokens += info.get("completion_tokens", 0)
                    total_tokens += info.get("total_tokens", 0)
                    total_time += info.get("time_taken", 0)
            except Exception as e:
                print(f"Error reading {info_file}: {e}")
        
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Total Completion Tokens: {total_completion_tokens}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Total Time Taken (sum of all tasks): {total_time:.2f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.runner.mcts_runner <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    runner = MCTSRunner(config=config_path)
    runner.run_all_tasks()
