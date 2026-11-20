from src.system.types import WorkerType, WorkerInput
from src.system.prompt_factory import get_worker_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import _build_hint_from_steps

class ValueExtractorWorker(BaseWorker):
    """Worker that extracts relevant values."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = 'Use to find the exact database values for entities mentioned in the question (e.g., mapping "CA" to "California"). This prevents empty results due to value mismatches.'
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)

        return get_worker_prompt(
            self.worker_type,
            QUESTION=input_data.question,
            HINT=hint,
            SCHEMA_CONTEXT=input_data.schema
        )
