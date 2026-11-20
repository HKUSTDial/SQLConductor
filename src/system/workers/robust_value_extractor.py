"""Extract column-value conditions from structured model responses."""
import re
from loguru import logger

from src.system.types import WorkerType, WorkerInput
from src.system.prompt_factory import get_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import _build_hint_from_steps


class RobustValueExtractorWorker(BaseWorker):
    """Identify database values relevant to a question."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = 'Use to find the exact database values for entities mentioned in the question (e.g., mapping "CA" to "California"). This prevents empty results due to value mismatches.'
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)

        return get_prompt(
            "robust_identify_column_values",
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
