"""MCTS actions implemented through the worker API."""

import asyncio
import copy
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from src.database.sql_execution import cached_execute_sql_with_timeout, is_valid_execution_result
from src.llm_call.openai_llm import call_openai
from src.system.llm_client import LLMResponse
from src.system.types import ReasoningSession, ReasoningStep, WorkerType, VALID_WORKER_TRANSITIONS_UNRESTRICTED
from src.system.workers.factory import WorkerFactory
from src.system.workers.utils import build_worker_input, update_session_state, _extract_sql_query_answer


class MCTSNodeType(Enum):
    ROOT = "root"
    REPHRASE_QUESTION = "rephrase_question"
    SCHEMA_SELECTION = "schema_selection"
    IDENTIFY_COLUMN_VALUES = "identify_column_values"
    IDENTIFY_COLUMN_FUNCTIONS = "identify_column_functions"
    SQL_REVISION = "sql_revision"
    SQL_GENERATION = "sql_generation"
    END = "end"


class MCTSWorkerClient:
    """LLM client for MCTS workers."""

    def __init__(self, llm_kwargs):
        self.llm_kwargs = dict(llm_kwargs)

    async def generate(self, prompt, **kwargs):
        # MCTS n controls analysis branches; each worker sets its own sampling budget.
        params = {**self.llm_kwargs, "n": 1, **kwargs}
        try:
            contents = await asyncio.to_thread(call_openai, prompt, **params)
            return LLMResponse(
                content=contents[0] if contents else "", contents=contents,
                model=params["model"],
            )
        except Exception as error:
            return LLMResponse(content="", model=params["model"], error=str(error))


def _worker_input(node):
    session = ReasoningSession(
        question=node.original_question, schema=node.schema_context,
        db_id=node.db_id, db_path=str(Path(node.db_root_dir) / node.db_id / f"{node.db_id}.sqlite"),
        evidence=node.hint, question_id=getattr(node, "question_id", None),
        retrieved_values=getattr(node, "retrieved_values", None),
    )
    steps = []
    for path_node in node.path_nodes:
        result = getattr(path_node, "worker_result", None)
        if result is None:
            continue
        update_session_state(session, result.worker_type, result)
        steps.append(ReasoningStep(
            turn=path_node.depth, llm_response="", worker_call=result.worker_type.value,
            worker_result=result,
        ))
    input_data = build_worker_input(session, node.table_schema_dict, steps)
    input_data.context["use_icl_examples"] = False
    return input_data


class MCTSAction:
    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        raise NotImplementedError()


class WorkerAction(MCTSAction):
    """Create search nodes from worker results."""

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        return asyncio.run(self._create_children_nodes(node, llm_kwargs))

    async def _create_children_nodes(self, node, llm_kwargs):
        worker = WorkerFactory.create_worker(self.worker_type)
        worker.config = replace(
            worker.config,
            temperature=llm_kwargs.get("temperature", worker.config.temperature),
            model_config=replace(
                worker.config.model_config,
                max_tokens=llm_kwargs.get("max_tokens", worker.config.model_config.max_tokens),
            ),
        )
        worker.llm_client = MCTSWorkerClient(llm_kwargs)
        is_sql_action = self.worker_type in (WorkerType.SQL_CONSTRUCTOR, WorkerType.SQL_REFINER)
        branch_count = 1 if is_sql_action else llm_kwargs.get("n", 1)
        children = []
        seen = set()
        for _ in range(branch_count):
            input_data = _worker_input(node)
            result = await worker.execute(input_data)
            if result.result in seen:
                continue
            seen.add(result.result)
            child = copy.deepcopy(node)
            child.node_type = self.node_type
            child.parent_node = node
            child.parent_action = self
            child.depth = node.depth + 1
            child.children = []
            child.N = child.Q = 0
            child.path_nodes = node.path_nodes + [child]
            child.worker_result = result
            setattr(child, self.result_field, result.result)
            if self.worker_type == WorkerType.SCHEMA_PRUNER:
                child.selected_schema_dict = None
            if is_sql_action:
                sql = _extract_sql_query_answer(result.result)
                child.is_valid_sql_query = bool(sql) and is_valid_execution_result(
                    cached_execute_sql_with_timeout(input_data.context["db_path"], sql)
                )
                child.consistency_score = (result.metadata or {}).get("confidence", 0.0)
            if child.is_terminal():
                # At the depth budget, return the current candidate without another worker call.
                session = _worker_input(child).context["session"]
                child.final_sql_query = session.current_sql
                child.consistency_score = session.sql_consistency_score
            children.append(child)
        return children


class RaphraseQuestionAction(WorkerAction):
    worker_type = WorkerType.QUESTION_DECOMPOSER
    node_type = MCTSNodeType.REPHRASE_QUESTION
    result_field = "rephrased_question"


class SchemaSelectionAction(WorkerAction):
    worker_type = WorkerType.SCHEMA_PRUNER
    node_type = MCTSNodeType.SCHEMA_SELECTION
    result_field = "selected_schema_context"


class IdentifyColumnValuesAction(WorkerAction):
    worker_type = WorkerType.VALUE_EXTRACTOR
    node_type = MCTSNodeType.IDENTIFY_COLUMN_VALUES
    result_field = "identified_column_values"


class IdentifyColumnFunctionsAction(WorkerAction):
    worker_type = WorkerType.OPERATION_IDENTIFIER
    node_type = MCTSNodeType.IDENTIFY_COLUMN_FUNCTIONS
    result_field = "identified_column_functions"


class SQLGenerationAction(WorkerAction):
    worker_type = WorkerType.SQL_CONSTRUCTOR
    node_type = MCTSNodeType.SQL_GENERATION
    result_field = "sql_query"


class SQLRevisionAction(WorkerAction):
    worker_type = WorkerType.SQL_REFINER
    node_type = MCTSNodeType.SQL_REVISION
    result_field = "revised_sql_query"


class EndAction(MCTSAction):
    """
    End the search.
    """
    worker_type = WorkerType.END
    node_type = MCTSNodeType.END

    def create_children_nodes(self, node: "MCTSNode", llm_kwargs: Dict[str, Any]) -> List["MCTSNode"]:
        child_node = copy.deepcopy(node)
        child_node.node_type = MCTSNodeType.END
        child_node.parent_node = node
        child_node.parent_action = self
        child_node.depth = node.depth + 1
        child_node.children = []
        child_node.N = child_node.Q = 0
        child_node.path_nodes = node.path_nodes + [child_node]
        child_node.worker_result = None
        
        session = _worker_input(node).context["session"]
        child_node.final_sql_query = session.current_sql
        child_node.consistency_score = session.sql_consistency_score
        return [child_node]


_WORKER_ACTIONS = {action.worker_type: action for action in (
    RaphraseQuestionAction, SchemaSelectionAction, IdentifyColumnValuesAction,
    IdentifyColumnFunctionsAction, SQLGenerationAction, SQLRevisionAction, EndAction,
)}

NODE_TYPE_TO_VALID_ACTIONS = {
    MCTSNodeType.ROOT: [_WORKER_ACTIONS[worker] for worker in VALID_WORKER_TRANSITIONS_UNRESTRICTED[None]],
    **{
        action.node_type: [_WORKER_ACTIONS[next_worker] for next_worker in VALID_WORKER_TRANSITIONS_UNRESTRICTED[worker]]
        for worker, action in _WORKER_ACTIONS.items()
    },
}
