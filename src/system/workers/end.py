from src.system.types import WorkerType, WorkerInput, WorkerResult
from src.system.workers.base import BaseWorker

class EndWorker(BaseWorker):
    """Terminal worker used to explicitly end reasoning."""

    def __init__(self, worker_type: WorkerType):
        """Initialize the EndWorker without requiring a configuration."""
        self.worker_type = worker_type
        self.name = worker_type.value
        self.description = "Use *only* when you have a valid, final SQL query that answers the user's question."
    
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        """This worker performs no action and returns success."""
        return WorkerResult(
            worker_type=self.worker_type,
            result="Reasoning process complete.",
            turn=0,
            success=True
        )
