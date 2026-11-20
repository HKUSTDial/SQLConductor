"""Identify operations from structured model responses."""
import re
from loguru import logger

from src.system.types import WorkerType, WorkerInput
from src.system.prompt_factory import get_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import _build_hint_from_steps


class RobustOperationIdentifierWorker(BaseWorker):
    """Identify the functions and operations required by a question."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = 'Use to clarify required mathematical or logical operations (e.g., "highest ratio", "percentage change") before writing SQL.'
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)
        
        return get_prompt(
            "robust_identify_column_functions",
            QUESTION=input_data.question,
            HINT=hint,
            SCHEMA_CONTEXT=input_data.schema
        )

    def _process_response(self, response: str, input_data: WorkerInput) -> str:
        """Return non-empty <answer> content, or the complete response otherwise."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
        logger.debug(f"[{self.worker_type.value}] No non-empty <answer> content; using the complete response.")
        return response
