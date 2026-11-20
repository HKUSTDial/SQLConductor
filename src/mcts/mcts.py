from src.mcts.mcts_node import *
from src.mcts.mcts_action import *
from src.mcts.reward import *
from src.runner.task import Task
import math
import random
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import pickle
from src.llm_call.openai_llm import DEFAULT_COST_RECORDER
from src.system.workers.utils import _build_and_condense_schema_context

class MCTSSolver:
    def __init__(self,
                 db_root_dir: str,
                 task: Task, 
                 max_rollout_steps: int,
                 max_depth: int,
                 exploration_constant: float,
                 save_root_dir: str,
                 llm_kwargs: Dict[str, Any],
                 reward_model: RewardModel):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.llm_kwargs = llm_kwargs
        self.reward_model = reward_model
        self.task = task
        self.db_root_dir = db_root_dir
        self.max_rollout_steps = max_rollout_steps
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.save_root_dir = save_root_dir
    
    def select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.is_terminal():
            if not all(child.N > 0 for child in current.children):
                return next(child for child in current.children if child.N == 0)
            
            current = max(current.children, key=lambda child: (child.Q / child.N) + self.exploration_constant * math.sqrt(math.log(current.N) / child.N))
        return current
    
    def expand(self, node: MCTSNode) -> List[MCTSNode]:
        if node.is_terminal():
            return
        assert node.children == [], f"Children nodes of node {node.node_type} before expansion is not empty"
        valid_action_space = get_valid_action_space_for_node(node)
        for action in valid_action_space:
            action_nodes = action.create_children_nodes(node, self.llm_kwargs)
            node.children.extend(action_nodes)
        
        random.shuffle(node.children)

    def simulate(self, node: MCTSNode) -> MCTSNode:
        assert node.children == [], f"Node before simulation have non-empty children"
        current = node
        while not current.is_terminal():
            self.expand(current)
            current = random.choice(current.children)
        return current

    def backpropagate(self, node: MCTSNode):
        current = node
        if current.N == 0:
            reward = self.reward_model.get_reward(current)
        else:
            reward = current.Q / current.N
        while current is not None:
            current.N += 1
            current.Q += reward
            current = current.parent_node
    
    def find_all_end_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        # Record the terminal state when the rollout budget is exhausted.
        if node.is_terminal():
            return [node]
        else:
            end_nodes = []
            for child in node.children:
                end_nodes.extend(self.find_all_end_nodes(child))
            return end_nodes
    
    def find_all_valid_reasoning_paths(self, node: MCTSNode) -> List[List[MCTSNode]]:
        end_nodes = self.find_all_end_nodes(node)
        reasoning_paths = []
        for end_node in end_nodes:
            reasoning_paths.append(end_node.path_nodes)
        return reasoning_paths
    
    def solve(self):
        start_time = time.time()
        start_prompt_tokens = DEFAULT_COST_RECORDER.get_total_prompt_tokens()
        start_completion_tokens = DEFAULT_COST_RECORDER.get_total_completion_tokens()
        
        schema_context = _build_and_condense_schema_context(self.task.table_schema_dict)
        root_node = MCTSNode(MCTSNodeType.ROOT,
                             parent_node=None,
                             parent_action=None,
                             depth=0,
                             db_id=self.task.db_id,
                             db_root_dir=self.db_root_dir,
                             original_question=self.task.question,
                             hint=self.task.evidence,
                             schema_context=schema_context,
                             table_schema_dict=self.task.table_schema_dict,
                             question_id=self.task.question_id,
                             retrieved_values=self.task.retrieved_values,
                             max_depth=self.max_depth)
        root_node.path_nodes = [root_node]
        
        for _ in range(self.max_rollout_steps):
            print(f"Question ID: {self.task.question_id}, Rollout step {_ + 1} / {self.max_rollout_steps}")
            leaf_node = self.select(root_node)
            if leaf_node.is_terminal():
                self.backpropagate(leaf_node)
                continue
            self.expand(leaf_node)
            leaf_node = random.choice(leaf_node.children)
            end_node = self.simulate(leaf_node)
            self.backpropagate(end_node)
        
        all_valid_reasoning_paths = self.find_all_valid_reasoning_paths(root_node)
        save_path = Path(self.save_root_dir) / f"{self.task.question_id}.pkl"
        print(f"Question ID: {self.task.question_id} done, Number of valid reasoning paths: {len(all_valid_reasoning_paths)}")
        
        end_time = time.time()
        time_taken = end_time - start_time
        prompt_tokens = DEFAULT_COST_RECORDER.get_total_prompt_tokens() - start_prompt_tokens
        completion_tokens = DEFAULT_COST_RECORDER.get_total_completion_tokens() - start_completion_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        info_dict = {
            "question_id": self.task.question_id,
            "time_taken": time_taken,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        info_save_path = Path(self.save_root_dir) / f"{self.task.question_id}_info.json"
        with open(info_save_path, "w") as f:
            json.dump(info_dict, f, indent=4)
            
        with open(save_path, "wb") as f:
            pickle.dump(all_valid_reasoning_paths, f)
