from src.system.types import WorkerType, WorkerInput
from src.system.prompt_factory import get_worker_prompt
from src.system.workers.base import BaseWorker

class QuestionDecomposerWorker(BaseWorker):
    """Worker that decomposes questions."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "Use when the question is long, convoluted, or contains multiple distinct requirements. It simplifies the question for other workers."
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        return get_worker_prompt(
            self.worker_type,
            QUESTION=input_data.question,
            HINT="",  # Rephrasing the question does not need a hint.
            SCHEMA_CONTEXT=input_data.schema
        )
