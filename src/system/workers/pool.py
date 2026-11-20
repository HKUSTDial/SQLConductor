from typing import Optional
from loguru import logger
from src.system.config import config
from src.system.types import WorkerType, WorkerInput, WorkerResult
from src.system.tools import ToolRegistry
from src.system.workers.base import BaseWorker
from src.system.workers.factory import WorkerFactory

class WorkerPool(ToolRegistry):
    """Manager for the worker pool."""
    
    def __init__(self):
        super().__init__()
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize all workers."""
        for worker_type in WorkerType:
            try:
                worker = WorkerFactory.create_worker(worker_type)
                self.register(worker)
                
                model_name = "N/A"
                if hasattr(worker, 'config') and worker.config and hasattr(worker.config, 'model_config') and worker.config.model_config:
                    model_name = worker.config.model_config.model_name
                
                logger.info(f"Worker '{worker_type.value}' initialized successfully: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize worker {worker_type.value}: {e}")
    
    def get_worker(self, worker_type: WorkerType) -> Optional[BaseWorker]:
        """Get a worker."""
        return self.get_tool(worker_type)
    
    async def execute_worker(self, worker_type: WorkerType, input_data: WorkerInput) -> WorkerResult:
        """Execute a worker task."""
        worker = self.get_tool(worker_type)
        if not worker:
            return WorkerResult(
                worker_type=worker_type,
                result="",
                turn=0,
                success=False,
                error_message=f"Worker {worker_type.value} not found"
            )
        
        return await worker.execute(input_data)
    
    def update_worker_model(self, worker_type: WorkerType, new_port: int):
        """Update the worker model port."""
        config.update_worker_model(worker_type.value, new_port)
        
        try:
            worker = WorkerFactory.create_worker(worker_type)
            self.register(worker)
            logger.info(f"Worker {worker_type.value} updated to port {new_port}")
        except Exception as e:
            logger.error(f"Failed to update worker {worker_type.value}: {e}")
