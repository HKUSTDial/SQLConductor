"""
Type definitions used by the system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class WorkerType(Enum):
    """Worker types."""
    QUESTION_DECOMPOSER = "Question Decomposer"
    SCHEMA_PRUNER = "Schema Pruner"
    VALUE_EXTRACTOR = "Value Extractor"
    OPERATION_IDENTIFIER = "Operation Identifier"
    SQL_CONSTRUCTOR = "SQL Constructor"
    SQL_REFINER = "SQL Refiner"
    END = "End"


FALLBACK_PIPELINE = [
    WorkerType.SCHEMA_PRUNER,
    WorkerType.SQL_CONSTRUCTOR,
    WorkerType.SQL_REFINER
]

FALLBACK_WORKER_OVERRIDES = {
    WorkerType.SCHEMA_PRUNER: "robust",
    WorkerType.SQL_CONSTRUCTOR: "robust",
    WorkerType.SQL_REFINER: "robust"
}


class ActionName(Enum):
    """Action names."""
    REPHRASE_QUESTION = "RaphraseQuestionAction"
    SCHEMA_SELECTION = "SchemaSelectionAction"
    IDENTIFY_COLUMN_VALUES = "IdentifyColumnValuesAction"
    IDENTIFY_COLUMN_FUNCTIONS = "IdentifyColumnFunctionsAction"
    SQL_GENERATION = "SQLGenerationAction"
    SQL_REVISION = "SQLRevisionAction"


ACTION_WORKER_MAP = {
    ActionName.REPHRASE_QUESTION: WorkerType.QUESTION_DECOMPOSER,
    ActionName.SCHEMA_SELECTION: WorkerType.SCHEMA_PRUNER,
    ActionName.IDENTIFY_COLUMN_VALUES: WorkerType.VALUE_EXTRACTOR,
    ActionName.IDENTIFY_COLUMN_FUNCTIONS: WorkerType.OPERATION_IDENTIFIER,
    ActionName.SQL_GENERATION: WorkerType.SQL_CONSTRUCTOR,
    ActionName.SQL_REVISION: WorkerType.SQL_REFINER
}


WORKER_TEMPLATE_MAP = {
    WorkerType.QUESTION_DECOMPOSER: "raphrase_question",
    WorkerType.SCHEMA_PRUNER: "schema_selection",
    WorkerType.VALUE_EXTRACTOR: "identify_column_values",
    WorkerType.OPERATION_IDENTIFIER: "identify_column_functions",
    WorkerType.SQL_CONSTRUCTOR: "sql_generation",
    WorkerType.SQL_REFINER: "sql_revision"
}


META_AGENT_SYSTEM_TEMPLATE = "meta_agent_system"
META_AGENT_USER_TEMPLATE = "meta_agent_user"


VALID_WORKER_TRANSITIONS = {
    None: [
        WorkerType.QUESTION_DECOMPOSER,
        WorkerType.SCHEMA_PRUNER,
        WorkerType.VALUE_EXTRACTOR,
        WorkerType.OPERATION_IDENTIFIER,
        WorkerType.SQL_CONSTRUCTOR,
    ],
    WorkerType.QUESTION_DECOMPOSER: [
        WorkerType.SCHEMA_PRUNER,
        WorkerType.VALUE_EXTRACTOR,
        WorkerType.OPERATION_IDENTIFIER,
        WorkerType.SQL_CONSTRUCTOR,
    ],
    WorkerType.SCHEMA_PRUNER: [
        WorkerType.VALUE_EXTRACTOR,
        WorkerType.OPERATION_IDENTIFIER,
        WorkerType.SQL_CONSTRUCTOR,
    ],
    WorkerType.VALUE_EXTRACTOR: [
        WorkerType.SCHEMA_PRUNER,
        WorkerType.OPERATION_IDENTIFIER,
        WorkerType.SQL_CONSTRUCTOR,
    ],
    WorkerType.OPERATION_IDENTIFIER: [
        WorkerType.SCHEMA_PRUNER,
        WorkerType.VALUE_EXTRACTOR,
        WorkerType.SQL_CONSTRUCTOR,
    ],
    WorkerType.SQL_CONSTRUCTOR: [
        WorkerType.SQL_REFINER,
        WorkerType.END,
    ],
    WorkerType.SQL_REFINER: [
        WorkerType.END,
    ],
    WorkerType.END: []
}

_ANALYSIS_WORKERS = [
    WorkerType.QUESTION_DECOMPOSER,
    WorkerType.SCHEMA_PRUNER,
    WorkerType.VALUE_EXTRACTOR,
    WorkerType.OPERATION_IDENTIFIER,
]

VALID_WORKER_TRANSITIONS_UNRESTRICTED = {
    None: [*_ANALYSIS_WORKERS, WorkerType.SQL_CONSTRUCTOR],
    WorkerType.QUESTION_DECOMPOSER: [*_ANALYSIS_WORKERS, WorkerType.SQL_CONSTRUCTOR],
    WorkerType.SCHEMA_PRUNER:       [*_ANALYSIS_WORKERS, WorkerType.SQL_CONSTRUCTOR],
    WorkerType.VALUE_EXTRACTOR:     [*_ANALYSIS_WORKERS, WorkerType.SQL_CONSTRUCTOR],
    WorkerType.OPERATION_IDENTIFIER:[*_ANALYSIS_WORKERS, WorkerType.SQL_CONSTRUCTOR],
    WorkerType.SQL_CONSTRUCTOR: [
        *_ANALYSIS_WORKERS,
        WorkerType.SQL_CONSTRUCTOR,
        WorkerType.SQL_REFINER,
        WorkerType.END,
    ],
    WorkerType.SQL_REFINER: [
        *_ANALYSIS_WORKERS,
        WorkerType.SQL_CONSTRUCTOR,
        WorkerType.SQL_REFINER,
        WorkerType.END,
    ],
    WorkerType.END: [],
}

@dataclass
class WorkerInput:
    """Worker input data."""
    question: str
    schema: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class SQLCandidate:
    """Represents a single SQL candidate with metadata for candidate list mode."""
    sql: str
    raw_content: str
    strategy: Optional[str] = None
    is_valid: bool = False  # Whether execution succeeded (no error or timeout; may still be empty).
    execution_time: Optional[float] = None  # Execution time in seconds.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "sql": self.sql,
            "is_valid": self.is_valid
        }


@dataclass
class WorkerResult:
    """Worker execution result."""
    worker_type: 'WorkerType'
    result: str
    turn: int
    success: bool
    error_message: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    sql_candidates: Optional[List['SQLCandidate']] = None

@dataclass
class ReasoningStep:
    """Reasoning step."""
    turn: int
    llm_response: str
    worker_call: Optional[str] = None
    worker_result: Optional[WorkerResult] = None
    reasoning: str = ""
    exclude_from_training: bool = False

@dataclass
class ReasoningSession:
    """State of a single task's reasoning session."""
    question: str
    schema: str
    db_id: str
    db_path: str
    evidence: Optional[str] = None
    question_id: Optional[str] = None
    retrieved_values: Optional[Dict[str, Dict[str, list]]] = None
    
    rephrased_question: Optional[str] = None
    selected_schema_context: Optional[str] = None
    identified_column_values: Optional[str] = None
    identified_column_functions: Optional[str] = None
    constructed_sql: Optional[str] = None
    refined_sql: Optional[str] = None
    
    sql_candidates: Optional[List['SQLCandidate']] = None
    refined_candidates: Optional[List['SQLCandidate']] = None
    sql_consistency_score: Optional[float] = None
    
    
    final_sql: Optional[str] = None
    success: bool = False
    
    total_tokens: int = 0
    meta_agent_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    worker_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    execution_time: float = 0.0
    
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None

    @property
    def current_question(self) -> str:
        """Return the latest version of the question."""
        return self.rephrased_question or self.question

    @property
    def current_schema(self) -> str:
        """Return the latest version of the database schema."""
        return self.selected_schema_context or self.schema

    @property
    def current_sql(self) -> Optional[str]:
        """Return the latest available SQL query."""
        return self.refined_sql or self.constructed_sql

@dataclass
class TaskResult:
    """
    Saves the complete result for each task, for serialization to a pkl file
    """
    task_index: int
    question_id: Any
    db_id: str
    difficulty: str
    question: str
    schema: str
    session_state: ReasoningSession
    reasoning_trace: List[ReasoningStep]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Converts to a summary dictionary format for final evaluation."""
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "difficulty": self.difficulty,
            "question": self.question,
            "final_sql": self.session_state.final_sql,
            "success": self.session_state.success,
            "total_turns": len(self.reasoning_trace),
            "total_tokens": self.session_state.total_tokens,
            "meta_agent_usage": self.session_state.meta_agent_usage,
            "worker_usage": self.session_state.worker_usage,
            "execution_time": self.session_state.execution_time,
            "fallback_triggered": getattr(self.session_state, "fallback_triggered", False),
            "fallback_reason": getattr(self.session_state, "fallback_reason", None)
        }

    def to_candidates_dict(self) -> Dict[str, Any]:
        """Convert candidate SQL information to a dictionary."""
        sql_candidates = getattr(self.session_state, 'sql_candidates', None) or []
        refined_candidates = getattr(self.session_state, 'refined_candidates', None) or []
        
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "difficulty": self.difficulty,
            "question": self.question,
            "evidence": getattr(self.session_state, 'evidence', None),
            "final_sql": self.session_state.final_sql,
            "success": self.session_state.success,
            "fallback_triggered": getattr(self.session_state, "fallback_triggered", False),
            "sql_candidates": {
                "count": len(sql_candidates),
                "valid_count": sum(1 for c in sql_candidates if c.is_valid),
                "candidates": [c.to_dict() for c in sql_candidates] if sql_candidates else []
            },
            "refined_candidates": {
                "count": len(refined_candidates),
                "valid_count": sum(1 for c in refined_candidates if c.is_valid),
                "candidates": [c.to_dict() for c in refined_candidates] if refined_candidates else []
            }
        }
