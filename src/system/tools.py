"""
Tools and Registry for Agent
"""
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
from loguru import logger

from src.system.types import WorkerType, WorkerInput, WorkerResult, VALID_WORKER_TRANSITIONS

class AgentTool(ABC):
    """Abstract base class for tools used by the Agent"""
    name: str = ""
    description: str = ""
    worker_type: WorkerType

    @abstractmethod
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        """
        Execute the tool.
        Input data contains injected context (session, db_schema, etc.)
        """
        pass

class ToolRegistry:
    """Registry for managing available tools"""
    def __init__(self):
        self._tools: Dict[WorkerType, AgentTool] = {}
        
    def register(self, tool: AgentTool):
        if tool.worker_type in self._tools:
            logger.warning(f"Overwriting tool for worker type: {tool.worker_type}")
        self._tools[tool.worker_type] = tool
        logger.debug(f"Registered tool: {tool.name} ({tool.worker_type.value})")
        
    def get_tool(self, worker_type: WorkerType) -> Optional[AgentTool]:
        return self._tools.get(worker_type)
        
    def get_all_tools(self) -> List[AgentTool]:
        return list(self._tools.values())

    def get_available_tools(
        self,
        current_worker_type: Optional[WorkerType],
        transitions: Optional[Dict] = None,
    ) -> List[AgentTool]:
        """
        Get valid tools based on the current state (last successful worker type).
        Accepts an optional transitions table; defaults to VALID_WORKER_TRANSITIONS.
        """
        transitions = transitions or VALID_WORKER_TRANSITIONS
        valid_next_types = transitions.get(current_worker_type)
        if valid_next_types is None:
            return []
            
        available_tools = []
        for worker_type in valid_next_types:
            tool = self.get_tool(worker_type)
            if tool:
                available_tools.append(tool)
        return available_tools


@dataclass
class ToolCall:
    """Represents a parsed tool call"""
    tool_name: str
    worker_type: WorkerType
    reasoning: str = ""

class OutputParser(ABC):
    """Abstract base class for parsing LLM output into ToolCalls"""
    
    @abstractmethod
    def parse(self, llm_response: str) -> Optional[ToolCall]:
        pass

    @abstractmethod
    def needs_tool_call(self, llm_response: str) -> bool:
        pass

class XMLOutputParser(OutputParser):
    """Parses XML-style tool calls (e.g. <call>ToolName</call>)"""
    
    def parse(self, llm_response: str) -> Optional[ToolCall]:
        worker_calls = re.findall(r'<call>(.*?)</call>', llm_response)
        if not worker_calls:
            return None
        
        worker_name = worker_calls[-1]
        
        for wt in WorkerType:
            if wt.value == worker_name:
                return ToolCall(tool_name=worker_name, worker_type=wt)
        
        logger.warning(f"Unknown worker in XML tag: {worker_name}")
        return None

    def needs_tool_call(self, llm_response: str) -> bool:
        worker_tags = [f"<call>{worker_type.value}</call>" for worker_type in WorkerType]
        return any(tag in llm_response for tag in worker_tags)
