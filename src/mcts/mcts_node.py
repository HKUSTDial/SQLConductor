from typing import List, Optional, Dict, Any
from src.mcts.mcts_action import *

def get_valid_action_space_for_node(node: "MCTSNode") -> List["MCTSAction"]:
    if node.is_terminal():
        return []
    # Failed workers do not advance the transition state.
    last_node = node
    while (result := getattr(last_node, "worker_result", None)) is not None and not result.success:
        last_node = last_node.parent_node
    return [action() for action in NODE_TYPE_TO_VALID_ACTIONS[last_node.node_type]]

class MCTSNode:
    def __init__(self,
                 node_type: "MCTSNodeType",
                 parent_node: Optional["MCTSNode"] = None,
                 parent_action: Optional["MCTSAction"] = None,
                 depth: int = 0,
                 db_id: str = "",
                 db_root_dir: str = "",
                 original_question: str = "",
                 hint: str = "",
                 schema_context: str = "",
                 table_schema_dict: Optional[Dict[str, "TableSchema"]] = None,
                 path_nodes: List["MCTSNode"] = [],
                 rephrased_question: Optional[str] = None,
                 selected_schema_dict: Optional[Dict[str, "TableSchema"]] = None,
                 selected_schema_context: Optional[str] = None,
                 identified_column_values: Optional[str] = None,
                 identified_column_functions: Optional[str] = None,
                 sql_query: Optional[str] = None,
                 revised_sql_query: Optional[str] = None,
                 final_sql_query: Optional[str] = None,
                 consistency_score: Optional[float] = None,
                 is_valid_sql_query: Optional[bool] = None,
                 llm_kwargs: Optional[Dict[str, Any]] = None,
                 question_id: Optional[str] = None,
                 retrieved_values: Optional[Dict[str, Dict[str, list]]] = None,
                 max_depth: Optional[int] = None
                 ):
        self.node_type = node_type
        self.parent_node = parent_node
        self.parent_action = parent_action
        self.depth = depth
        self.db_id = db_id
        self.db_root_dir = db_root_dir
        self.original_question = original_question
        self.hint = hint
        self.schema_context = schema_context
        self.table_schema_dict = table_schema_dict
        self.children : List[MCTSNode] = []
        self.path_nodes = path_nodes
        
        self.rephrased_question = rephrased_question
        self.selected_schema_dict = selected_schema_dict
        self.selected_schema_context = selected_schema_context
        self.identified_column_values = identified_column_values
        self.identified_column_functions = identified_column_functions
        self.sql_query = sql_query
        self.revised_sql_query = revised_sql_query
        self.final_sql_query = final_sql_query
        self.consistency_score = consistency_score
        self.is_valid_sql_query = is_valid_sql_query
        self.llm_kwargs = llm_kwargs
        self.question_id = question_id
        self.retrieved_values = retrieved_values
        self.worker_result = None
        self.max_depth = max_depth

        self.Q = 0
        self.N = 0
    
    def create_children(self):
        if self.children or self.is_terminal():
            return
        
        valid_action_space = get_valid_action_space_for_node(self)
        for action in valid_action_space:
            self.children.extend(action.create_children_nodes(self, self.llm_kwargs))
            
    def is_terminal(self):
        max_depth = getattr(self, "max_depth", None)
        return self.node_type == MCTSNodeType.END or (max_depth is not None and self.depth >= max_depth)
