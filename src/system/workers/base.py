import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from src.system.config import config
from src.system.llm_client import get_llm_client, LLMResponse
from src.system.types import WorkerType, WorkerResult, WorkerInput
from src.system.tools import AgentTool


class BaseWorker(AgentTool):
    """Base class for workers."""
    
    def __init__(self, worker_type: WorkerType):
        self.worker_type = worker_type
        self.name = worker_type.value
        self.description = f"Worker for {worker_type.value}"
        self.config = config.get_worker_config(worker_type.value)
        if not self.config:
            raise ValueError(f"No configuration found for worker: {worker_type.value}")
        
        self.llm_client = get_llm_client(self.config.model_config)
        self.turn_count = 0
    
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        """Execute a worker task."""
        self.turn_count += 1
        
        try:
            prompt = self._build_prompt(input_data)
            
            response = await self._call_llm(prompt)
            
            if response.error:
                return WorkerResult(
                    worker_type=self.worker_type,
                    result="",
                    turn=self.turn_count,
                    success=False,
                    error_message=response.error
                )
            
            result = self._process_response(response.content, input_data)
            
            return WorkerResult(
                worker_type=self.worker_type,
                result=result,
                turn=self.turn_count,
                success=True,
                usage=response.usage
            )
            
        except Exception as e:
            logger.error(f"Worker {self.worker_type.value} execution failed: {e}")
            return WorkerResult(
                worker_type=self.worker_type,
                result="",
                turn=self.turn_count,
                success=False,
                error_message=str(e)
            )
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        """Build the prompt. Subclasses must override this."""
        raise NotImplementedError
    
    def _process_response(self, response: str, input_data: WorkerInput) -> str:
        """Process the response. Subclasses may override this."""
        return response
    
    async def _call_llm(self, prompt: str) -> LLMResponse:
        """Call the LLM."""
        return await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.model_config.max_tokens,
            stop=self.config.model_config.stop
        )
