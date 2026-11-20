
import os
# Set distributed timeouts before importing torch to accommodate long rollouts.
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "10800"
os.environ["NCCL_TIMEOUT"] = "10800"                      # NCCL library timeout (seconds)
# DEEPSPEED_TIMEOUT is in minutes, controls deepspeed's default_pg_timeout
os.environ["DEEPSPEED_TIMEOUT"] = "180"

import torch
# Apply the timeout to secondary process groups created by DeepSpeed as well.
import torch.distributed.distributed_c10d as _c10d
from datetime import timedelta as _timedelta
_c10d.default_pg_timeout = _timedelta(hours=3)
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
from accelerate import Accelerator
from accelerate.utils import DistributedType, InitProcessGroupKwargs
from peft import LoraConfig, get_peft_model, TaskType
from datetime import timedelta

from src.train.meta_agent_rl.model_utils import load_model_and_tokenizer, LocalLLMClient
from src.train.meta_agent_rl.dataset import RLDataset
from src.system.config import ModelConfig, config
from src.system.meta_agent import MetaAgent
from src.system.types import WorkerType, ReasoningStep
from src.train.meta_agent_rl.rewards import (
    compute_outcome_reward,
    compute_executability_reward,
)
from src.database.schema import DatabaseSchema, TableSchema


def _disable_policy_dropout(model):
    """Disable dropout in policy modules."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


class TrainableMetaAgent(MetaAgent):
    """MetaAgent with a local policy client and per-turn prompt capture for RL."""
    def __init__(self, local_client):
        self.config = config.get_meta_agent_config().copy()
        self.max_turns = self.config.get("max_turns", 10)
        self.llm_client = local_client
        self.step_prompts: List[str] = []
        
        from src.system.workers import WorkerPool
        self.worker_pool = WorkerPool()
        
        from src.system.tools import XMLOutputParser
        self.output_parser = XMLOutputParser()

        from src.system.types import VALID_WORKER_TRANSITIONS, VALID_WORKER_TRANSITIONS_UNRESTRICTED
        self.unrestricted = True
        self.worker_transitions = (
            VALID_WORKER_TRANSITIONS_UNRESTRICTED if self.unrestricted else VALID_WORKER_TRANSITIONS
        )
        
        all_tools = self.worker_pool.get_all_tools()
        worker_descriptions_lines = []
        for tool in all_tools:
            worker_descriptions_lines.append(f"*   **{tool.name}**: {tool.description}")
        worker_descriptions_block = "\n".join(worker_descriptions_lines)
        
        from src.system.prompt_factory import get_meta_agent_system_prompt
        self.system_prompt = get_meta_agent_system_prompt(worker_descriptions=worker_descriptions_block)
        
        self.fallback_workers: Dict[WorkerType, Any] = {}
        
    async def _call_llm_for_reasoning(self, user_prompt: str):
        """Intercept LLM call to capture the prompt."""
        self.step_prompts.append(user_prompt)
        return await super()._call_llm_for_reasoning(user_prompt)
        
class GRPOTrainer:
    """
    GRPO trainer for meta-agent RL training.
    
    Args:
        model_name_or_path: Path to base model or HuggingFace model ID
        train_tasks_path: Path to training tasks pickle file
        trajectory_summary_path: Path to the trajectory summary used to select training tasks
        output_dir: Output directory for checkpoints
        learning_rate: Learning rate for optimizer
        batch_size: Batch size per GPU (number of questions per batch)
        group_size: Number of trajectories to sample per question (G in GRPO)
        max_epochs: Number of training epochs
        beta: KL divergence coefficient
        rollout_timeout: Timeout in seconds for single trajectory rollout
        lora_config: Dict with LoRA configuration (enabled, r, lora_alpha, lora_dropout, target_modules)
        quantization_config: Dict with quantization settings (enabled, load_in_8bit, llm_int8_skip_modules)
        checkpoint_config: Dict with checkpoint settings (save_every_steps, max_checkpoints, gated_best, kl_warning_threshold)
        serial_group_processing: If True, process samples serially with gradient accumulation
        reward_lambda: Outcome reward weight; executability receives weight (1 - reward_lambda)
        curriculum_threshold: Retain questions with an MCTS success ratio below this value
    """
    def __init__(
        self,
        model_name_or_path: str,
        train_tasks_path: str,
        trajectory_summary_path: str,
        output_dir: str,
        learning_rate: float = 1e-6,
        batch_size: int = 1, 
        group_size: int = 4, 
        max_epochs: int = 1,
        beta: float = 0.1, 
        rollout_timeout: int = 600,
        lora_config: dict = None,
        quantization_config: dict = None,
        checkpoint_config: dict = None,
        serial_group_processing: bool = False,
        reward_lambda: float = 0.8,
        curriculum_threshold: Optional[float] = None
    ):
        if not 0.0 <= reward_lambda <= 1.0:
            raise ValueError("reward_lambda must be between 0 and 1")
        self.reward_lambda = reward_lambda

        self.dataset = RLDataset(train_tasks_path, trajectory_summary_path, curriculum_threshold)
        if not self.dataset:
            raise ValueError("No RL training tasks selected; check the task IDs, summary, and curriculum_threshold")

        # Collective timeout must exceed per-trajectory rollout timeouts.
        kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            kwargs_handlers=[kwargs],
            log_with="tensorboard",
            project_dir=output_dir
        )
        self.device = self.accelerator.device
        
        self.rollout_timeout_sec = rollout_timeout
        
        self.output_dir = output_dir
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name_or_path, 
            device=None, 
            allow_zero3_init=True,
            quantization_config=quantization_config
        )

        if lora_config is None:
            lora_config = {'enabled': True, 'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05, 
                          'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]}
        
        if lora_config.get('enabled', True):
            logger.info("Applying LoRA to model...")
            logger.info(f"LoRA Config: r={lora_config.get('r', 16)}, alpha={lora_config.get('lora_alpha', 32)}, "
                       f"dropout={lora_config.get('lora_dropout', 0.05)}")
            
            # Register input-gradient hooks before applying PEFT.
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            
            self.model.config.use_cache = False
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_config.get('r', 16), 
                lora_alpha=lora_config.get('lora_alpha', 32), 
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
            logger.info("Gradient Checkpointing enabled and LoRA applied.")
            
            trainable_params = 0
            all_param = 0
            for name, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    if "lora" not in name and "modules_to_save" not in name:
                         logger.warning(f"Unexpected trainable parameter: {name}")
            
            logger.info(f"Trainable parameters: {trainable_params}; total parameters: {all_param}; percentage: {100 * trainable_params / all_param:.4f}")

        # Keep dropout disabled during rollout and loss evaluation.
        _disable_policy_dropout(self.model)

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
                "weight_decay": 0.01,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self.dataset.collate_fn
        )
        
        self.group_size = group_size
        self.max_epochs = max_epochs
        self.beta = beta
        
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        
        client_config = ModelConfig(base_url="", api_key="", model_name="local-policy")
        self.local_client = LocalLLMClient(client_config, self.model, self.tokenizer)
        
        self.agent = TrainableMetaAgent(self.local_client)
        
        if checkpoint_config is None:
            checkpoint_config = {
                'save_every_steps': 50,
                'max_checkpoints': 3,
                'gated_best': {'enabled': True, 'kl_threshold': 0.05, 'use_reward_ema': True, 'ema_alpha': 0.1},
                'kl_warning_threshold': 0.1
            }
        self.checkpoint_config = checkpoint_config
        
        self.global_step = 0
        self.saved_checkpoints = []  # List of checkpoint paths (oldest first)
        self.best_reward = float('-inf')
        self.reward_ema = None
        
        self.serial_group_processing = serial_group_processing
        if serial_group_processing:
            logger.info("Serial group processing enabled.")
        
    def _load_database_schema(self, db_id: str, table_schema_dict: Dict[str, Any]) -> DatabaseSchema:
        """
        Load Schema from provided dict.
        """
        from src.system.config import config
        database_root_dir = config.config_data.get('system', {}).get('data', {}).get('database_root_dir', 'data/bird/dev/dev_databases')
        
        tables = {}
        if table_schema_dict:
            for table_name, table_info in table_schema_dict.items():
                if isinstance(table_info, dict):
                    if "table_name" not in table_info:
                        table_info["table_name"] = table_name
                    tables[table_name] = TableSchema.from_table_schema_dict(table_info)
        
        return DatabaseSchema(db_id=db_id, db_directory=f"{database_root_dir}/{db_id}", tables=tables)

    def train(self):
        logger.info(f"Starting GRPO Training | Epochs: {self.max_epochs} | Group Size: {self.group_size}")
        
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("rl_training_logs", config={
                "epochs": self.max_epochs,
                "batch_size_per_device": self.dataloader.batch_size,
                "group_size": self.group_size,
                "reward_lambda": self.reward_lambda,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            self.csv_log_path = os.path.join(self.output_dir, "training_log.csv")
            file_exists = os.path.exists(self.csv_log_path)
            with open(self.csv_log_path, "a") as f:
                if not file_exists:
                    f.write("step,epoch,loss,reward_total,reward_outcome,reward_executability,kl,lr\n")
        
        if hasattr(self.model, "peft_config"):
            logger.info(f"Training with PEFT/LoRA active.")
        else:
            logger.warning("Training without PEFT/LoRA.")
        self.model.train()
        
        epoch_bar = tqdm(range(self.max_epochs), desc="Epochs", position=0, leave=True, disable=not self.accelerator.is_main_process)
        
        for epoch in epoch_bar:
            batch_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1} Training", position=1, leave=False, disable=not self.accelerator.is_main_process)
            
            epoch_loss = 0.0
            epoch_rewards = {"outcome": 0.0, "executability": 0.0, "total": 0.0}
            total_samples = 0
            
            for batch in batch_bar:
                question_ids = batch["question_id"]
                questions = batch["question"]
                db_ids = batch["db_id"]
                
                batch_loss = 0.0
                batch_rewards = {"outcome": 0.0, "executability": 0.0, "total": 0.0}
                
                all_step_samples = []
                
                for q_idx, q_text in enumerate(questions):
                    db_id = db_ids[q_idx]
                    
                    question_id = question_ids[q_idx]
                    ground_truth_sql = batch["ground_truth_sql"][q_idx]
                    evidence = batch["evidence"][q_idx]
                    table_schema_dict = batch["table_schema_dict"][q_idx]
                    trajectories = self.rollout(question_id, q_text, db_id, ground_truth_sql, evidence, table_schema_dict)
                    if not trajectories:
                        continue

                    if len(trajectories) < 2:
                        logger.warning(f"Skipping question [QID={question_id}] due to insufficient trajectories ({len(trajectories)})")
                        continue

                    # Normalize the combined reward within each question's group.
                    outcome_rewards = torch.tensor([t["rewards"]["outcome"] for t in trajectories], device=self.device)
                    executability_rewards = torch.tensor([t["rewards"]["executability"] for t in trajectories], device=self.device)
                    total_rewards = torch.tensor([t["rewards"]["total"] for t in trajectories], device=self.device)

                    batch_rewards["outcome"] += outcome_rewards.mean().item()
                    batch_rewards["executability"] += executability_rewards.mean().item()
                    batch_rewards["total"] += total_rewards.mean().item()
                    
                    reward_std, reward_mean = torch.std_mean(total_rewards, unbiased=False)
                    advantages = (total_rewards - reward_mean) / (reward_std + 1e-8)
                    
                    for i, traj in enumerate(trajectories):
                        steps = traj["steps"]
                        prompts = traj["prompts"]
                        
                        traj_advantage = advantages[i]
                        samples = self._tensorize_trajectory(steps, prompts, traj_advantage)
                        for sample in samples:
                            # Average trajectories within each question, then questions.
                            sample["token_weight"] /= len(trajectories) * len(questions)
                        all_step_samples.extend(samples)
                
                # Synchronize ranks before deciding whether to skip the batch.
                self.accelerator.wait_for_everyone()

                num_samples = len(all_step_samples)

                # Every rank must perform the same number of backward passes.
                local_count = torch.tensor([num_samples], device=self.device)
                global_max_count = int(self.accelerator.gather(local_count).max().item())

                if global_max_count == 0:
                    continue

                if num_samples == 0:
                    logger.warning(
                        f"Empty batch on this rank; {global_max_count} "
                        f"synchronization iterations."
                    )

                self.model.train()
                self.accelerator.unwrap_model(self.model).config.use_cache = False

                input_ids_list = [s["input_ids"] for s in all_step_samples]
                attention_mask_list = [s["attention_mask"] for s in all_step_samples]
                loss_mask_list = [s["loss_mask"] for s in all_step_samples]
                advantages_list = [s["advantage"] for s in all_step_samples]
                token_weights = torch.tensor(
                    [s["token_weight"] for s in all_step_samples], device=self.device,
                    dtype=torch.float32,
                )

                max_len = max((len(ids) for ids in input_ids_list), default=0)
                is_deepspeed = self.accelerator.distributed_type == DistributedType.DEEPSPEED
                backward_kwargs = {"scale_wrt_gas": False} if is_deepspeed else {}

                try:
                    if self.serial_group_processing:
                        
                        self.optimizer.zero_grad()
                        accumulated_loss = 0.0
                        accumulated_kl = 0.0
                        
                        for sample_idx in range(global_max_count):
                            if is_deepspeed:
                                self.model.set_gradient_accumulation_boundary(sample_idx == global_max_count - 1)
                            if sample_idx < num_samples:
                                single_ids = input_ids_list[sample_idx].unsqueeze(0)  # [1, L]
                                single_mask = attention_mask_list[sample_idx].unsqueeze(0)
                                single_loss_mask = loss_mask_list[sample_idx].unsqueeze(0)
                                single_advantage = advantages_list[sample_idx].unsqueeze(0)  # [1]
                                
                                loss = self.compute_loss(
                                    single_ids, single_mask, single_loss_mask, single_advantage,
                                    token_weights=token_weights[sample_idx:sample_idx + 1],
                                )
                                self.accelerator.backward(loss, **backward_kwargs)
                                
                                accumulated_loss += loss.item()
                                accumulated_kl += self._last_kl
                                
                                del single_ids, single_mask, single_loss_mask, single_advantage, loss
                            else:
                                # Keep backward calls synchronized across ranks.
                                dummy_ids = torch.tensor([[0]], device=self.device)
                                dummy_mask = torch.tensor([[1]], device=self.device)
                                dummy_out = self.model(input_ids=dummy_ids, attention_mask=dummy_mask)
                                dummy_loss = dummy_out.logits.sum() * 0.0
                                self.accelerator.backward(dummy_loss, **backward_kwargs)
                                del dummy_ids, dummy_mask, dummy_out, dummy_loss
                            
                            torch.cuda.empty_cache()
                        
                        batch_loss = accumulated_loss
                        self._last_kl = accumulated_kl
                        self.optimizer.step()
                    else:
                        if is_deepspeed:
                            self.model.set_gradient_accumulation_boundary(True)
                        if num_samples == 0:
                            dummy_ids = torch.tensor([[0]], device=self.device)
                            dummy_mask = torch.tensor([[1]], device=self.device)
                            dummy_out = self.model(input_ids=dummy_ids, attention_mask=dummy_mask)
                            dummy_loss = dummy_out.logits.sum() * 0.0
                            self.optimizer.zero_grad()
                            self.accelerator.backward(dummy_loss, **backward_kwargs)
                            self.optimizer.step()
                            del dummy_ids, dummy_mask, dummy_out, dummy_loss
                            torch.cuda.empty_cache()
                            batch_loss = 0.0
                        else:
                            padded_input_ids = []
                            padded_attention_mask = []
                            padded_loss_mask = []

                            for i in range(num_samples):
                                curr_len = len(input_ids_list[i])
                                pad_len = max_len - curr_len
                                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

                                ids = torch.cat([input_ids_list[i], torch.full((pad_len,), pad_id, device=self.device)])
                                mask = torch.cat([attention_mask_list[i], torch.zeros(pad_len, device=self.device)])
                                l_mask = torch.cat([loss_mask_list[i], torch.zeros(pad_len, device=self.device)])

                                padded_input_ids.append(ids)
                                padded_attention_mask.append(mask)
                                padded_loss_mask.append(l_mask)

                            input_embeddings = torch.stack(padded_input_ids).long()
                            attention_mask = torch.stack(padded_attention_mask).long()
                            loss_mask = torch.stack(padded_loss_mask).float()
                            batch_advantages = torch.stack(advantages_list)

                            loss = self.compute_loss(
                                input_embeddings, attention_mask, loss_mask, batch_advantages,
                                token_weights=token_weights,
                            )
                            batch_loss = loss.item()

                            self.optimizer.zero_grad()
                            self.accelerator.backward(loss, **backward_kwargs)
                            self.optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"Out of memory at step {self.global_step + 1} "
                        f"(Batch Size: {len(questions)}, Max Len: {max_len})."
                    )
                    raise
                
                self.global_step += 1
                
                current_reward = batch_rewards['total'] / (len(questions) + 1e-8)
                
                gated_cfg = self.checkpoint_config.get('gated_best', {})
                if gated_cfg.get('use_reward_ema', True):
                    alpha = gated_cfg.get('ema_alpha', 0.1)
                    if self.reward_ema is None:
                        self.reward_ema = current_reward
                    else:
                        self.reward_ema = alpha * current_reward + (1 - alpha) * self.reward_ema
                else:
                    self.reward_ema = current_reward
                
                current_kl = getattr(self, '_last_kl', 0.0)
                
                kl_warning_threshold = self.checkpoint_config.get('kl_warning_threshold', 0.1)
                if current_kl > kl_warning_threshold:
                    logger.warning(f"KL divergence ({current_kl:.4f}) exceeds threshold ({kl_warning_threshold})! "
                                  f"Training may be unstable. Consider reducing learning rate or increasing beta.")
                
                save_every_steps = self.checkpoint_config.get('save_every_steps', 50)
                if self.global_step % save_every_steps == 0:
                    self._save_step_checkpoint(self.global_step, current_kl, self.reward_ema)
                
                epoch_loss += batch_loss
                for k in epoch_rewards:
                    epoch_rewards[k] += batch_rewards[k]
                total_samples += len(questions)
                    
                n_qs = len(questions) + 1e-8
                batch_bar.set_postfix({
                    "step": f"{self.global_step}",
                    "loss": f"{batch_loss:.4f}",
                    "OR": f"{batch_rewards['outcome']/n_qs:.2f}",
                    "ER": f"{batch_rewards['executability']/n_qs:.2f}",
                    "kl": f"{current_kl:.4f}"
                })
                
                if self.accelerator.is_main_process:
                    metrics = {
                        "loss/batch": batch_loss,
                        "reward/total": current_reward.item() if torch.is_tensor(current_reward) else current_reward,
                        "reward/outcome": batch_rewards['outcome']/n_qs,
                        "reward/executability": batch_rewards['executability']/n_qs,
                        "kl_divergence": current_kl,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1
                    }
                    self.accelerator.log(metrics, step=self.global_step)
                    
                    with open(self.csv_log_path, "a") as f:
                        f.write(f"{self.global_step},{epoch+1},{metrics['loss/batch']:.4f},{metrics['reward/total']:.4f},"
                                f"{metrics['reward/outcome']:.4f},{metrics['reward/executability']:.4f},"
                                f"{metrics['kl_divergence']:.4f},{metrics['learning_rate']:.2e}\n")
            
            n_samples = total_samples + 1e-8
            avg_loss = epoch_loss / len(self.dataloader)
            avg_rewards = {k: v / n_samples for k, v in epoch_rewards.items()}
            
            if self.accelerator.is_main_process:
                logger.success(f"Epoch {epoch + 1} Finished | Steps: {self.global_step} | Loss: {avg_loss:.4f} | "
                               f"Rewards: OR={avg_rewards['outcome']:.3f}, "
                               f"ER={avg_rewards['executability']:.3f}, "
                               f"Total={avg_rewards['total']:.3f}")

    def rollout(self, question_id: str, question: str, db_id: str, ground_truth_sql: str, evidence: str, table_schema_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates G trajectories.
        Returns list of results dicts.
        """
        results = []
        
        self.model.eval()
        
        db_schema = self._load_database_schema(db_id, table_schema_dict)
        
        from src.system.config import config
        database_root_dir = config.config_data.get('system', {}).get('data', {}).get('database_root_dir', 'data/bird/dev/dev_databases')
        db_path = f"{database_root_dir}/{db_id}/{db_id}.sqlite"

        async def _run_episode():
            return await self.agent.reason(question, "", evidence, db_schema, db_path, question_id=question_id)

        async def _run_with_timeout():
             return await asyncio.wait_for(_run_episode(), timeout=self.rollout_timeout_sec)

        for _ in range(self.group_size):
            try:
                self.agent.step_prompts = []
                
                session, steps = asyncio.run(_run_with_timeout())
                
                prompts = list(self.agent.step_prompts)
                
                outcome_r = compute_outcome_reward(session.final_sql, ground_truth_sql, db_path)
                executability_r = compute_executability_reward(session.final_sql, db_path)
                rewards_dict = {
                    "outcome": outcome_r,
                    "executability": executability_r,
                    "total": self.reward_lambda * outcome_r + (1.0 - self.reward_lambda) * executability_r,
                }
                
                results.append({
                    "steps": steps,
                    "prompts": prompts,
                    "rewards": rewards_dict
                })
            except asyncio.TimeoutError:
                logger.warning(
                    f"Rollout timed out after {self.rollout_timeout_sec}s "
                    f"[QID={question_id}, db={db_id}]; trajectory skipped."
                )
                continue
            except Exception as e:
                logger.error(f"Rollout failed with error: {e}")
                continue
                
        return results

    def _tensorize_trajectory(self, steps: List[ReasoningStep], prompts: List[str], advantage: torch.Tensor) -> List[Dict]:
        """Convert captured rollout prompts and responses into per-step training samples."""
        samples = []
        
        # Captured prompts correspond only to policy-generated steps.
        
        prompt_idx = 0
        for step in steps:
            if getattr(step, "exclude_from_training", False):
                continue
            
            if prompt_idx >= len(prompts):
                break
                
            current_prompt = prompts[prompt_idx]
            response_text = step.llm_response
            
            messages = [
                {"role": "system", "content": self.agent.system_prompt},
                {"role": "user", "content": current_prompt}
            ]
            
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            
            response_ids = self.tokenizer.encode(response_text + self.tokenizer.eos_token, add_special_tokens=False)
            
            full_ids = prompt_ids + response_ids
            full_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            
            input_ids = torch.tensor(full_ids, dtype=torch.long, device=self.device)
            loss_mask = torch.tensor(full_mask, dtype=torch.float, device=self.device)
            
            samples.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "loss_mask": loss_mask,
                "advantage": advantage
            })
            
            prompt_idx += 1
            
        # Normalize by the rollout's shifted, supervised token count.
        total_tokens = sum(sample["loss_mask"][1:].sum().item() for sample in samples)
        if total_tokens == 0:
            return []
        for sample in samples:
            sample["token_weight"] = 1.0 / total_tokens

        return samples

    def compute_log_probs(self, model, input_ids, attention_mask, loss_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_loss_mask = loss_mask[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        nll = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        
        token_log_probs = -nll.view(shift_labels.size())
        
        del logits
        del shift_logits
        del flat_logits
        
        token_log_probs = token_log_probs * shift_loss_mask
        
        return token_log_probs, shift_loss_mask

    def compute_loss(self, input_ids, attention_mask, loss_mask, advantages, token_weights=None):
        """
        GRPO loss for one optimizer update per fresh batch of rollouts.

        token_weights contains each row's per-token coefficient: 1 / (B * G * N_tau).
        Without explicit weights, average the per-row token losses.
        """
        new_log_probs, active_mask = self.compute_log_probs(self.model, input_ids, attention_mask, loss_mask)
        # Evaluate exponentials in float32 even when the policy runs in fp16/bf16.
        new_log_probs = new_log_probs.float()
        active_mask = active_mask.to(new_log_probs.dtype)

        ratio = torch.exp(new_log_probs - new_log_probs.detach())
        pg_loss = -ratio * advantages.detach().float().unsqueeze(-1)

        per_token_kl = torch.zeros_like(new_log_probs)
        if self.beta != 0.0:
            unwrapped_for_peft = self.accelerator.unwrap_model(self.model)
            if not hasattr(unwrapped_for_peft, "disable_adapter"):
                raise ValueError(
                    "KL regularization requires a frozen reference policy. "
                    "Enable LoRA on the merged SFT checkpoint, or set beta=0."
                )
            # The frozen base is the merged SFT policy; only the RL adapter is disabled.
            with torch.no_grad(), unwrapped_for_peft.disable_adapter():
                ref_log_probs, _ = self.compute_log_probs(self.model, input_ids, attention_mask, loss_mask)

            log_ratio = ref_log_probs.detach().float() - new_log_probs
            log_ratio = torch.where(active_mask.bool(), log_ratio, 0.0)
            per_token_kl = torch.expm1(log_ratio) - log_ratio

        if token_weights is None:
            token_weights = active_mask.sum(dim=-1).clamp_min(1.0).reciprocal() / active_mask.size(0)
        weighted_mask = active_mask * token_weights.detach().to(active_mask).unsqueeze(-1)
        per_token_loss = pg_loss + self.beta * per_token_kl
        loss = (per_token_loss * weighted_mask).sum()
        self._last_kl = (per_token_kl.detach() * weighted_mask).sum().item()

        return loss

    def _save_step_checkpoint(self, step: int, current_kl: float, current_reward: float):
        """
        Save checkpoint at a given step with gated-best logic and rolling deletion.
        
        Args:
            step: Current global step number
            current_kl: Current KL divergence value
            current_reward: Current reward (or reward EMA)
        """
        self.accelerator.wait_for_everyone()
        
        if not self.accelerator.is_main_process:
            return
        
        gated_cfg = self.checkpoint_config.get('gated_best', {})
        max_checkpoints = self.checkpoint_config.get('max_checkpoints', 3)
        save_every_steps = self.checkpoint_config.get('save_every_steps', 10)
        
        if step == save_every_steps:
            first_save_path = os.path.join(self.output_dir, "checkpoint-first")
            self._save_model_to_path(first_save_path)
            logger.info(f"Saved first checkpoint at step {step} to {first_save_path}")
        
        step_save_path = os.path.join(self.output_dir, f"checkpoint-step-{step}")
        self._save_model_to_path(step_save_path)
        logger.info(f"Saved checkpoint at step {step} to {step_save_path}")
        
        self.saved_checkpoints.append(step_save_path)
        
        while len(self.saved_checkpoints) > max_checkpoints:
            old_path = self.saved_checkpoints.pop(0)
            if os.path.exists(old_path):
                import shutil
                shutil.rmtree(old_path)
                logger.info(f"Deleted checkpoint: {old_path}")
        
        if gated_cfg.get('enabled', True):
            kl_threshold = gated_cfg.get('kl_threshold', 0.05)
            
            if current_kl <= kl_threshold:
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    best_save_path = os.path.join(self.output_dir, "checkpoint-best")
                    self._save_model_to_path(best_save_path)
                    logger.success(f"Best checkpoint saved. Reward: {current_reward:.4f}, KL: {current_kl:.4f}")
            else:
                logger.debug(f"Skipping best checkpoint update: KL ({current_kl:.4f}) > threshold ({kl_threshold})")
    
    def _save_model_to_path(self, save_path: str):
        """Save model and tokenizer to the specified path."""
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        state_dict = self.accelerator.get_state_dict(self.model)
        unwrapped_model.save_pretrained(save_path, state_dict=state_dict)
        self.tokenizer.save_pretrained(save_path)
