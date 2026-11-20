from src.mcts.mcts_node import MCTSNode
from typing import Dict, Any

class RewardModel:
    def __init__(self, **kwargs):
        pass
    
    def get_reward(self, end_node: MCTSNode) -> float:
        pass

class MajorityVoteRewardModel(RewardModel):
    def __init__(self, llm_kwargs: Dict[str, Any]):
        self.llm_kwargs = llm_kwargs
        
    def get_reward(self, end_node: MCTSNode) -> float:
        assert end_node.is_terminal()
        if not end_node.final_sql_query:
            return 0.0
        return end_node.consistency_score or 0.0
