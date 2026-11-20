"""
Reasoning engine
Integrates MetaAgent, the worker pool, and configuration management
"""

import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time

from src.system.config import config
from src.system.meta_agent import MetaAgent
from src.system.workers import WorkerType
from src.system.types import ReasoningSession, ReasoningStep, WorkerResult
from src.database.schema import DatabaseSchema


class ReasoningEngine:
    """Main reasoning engine class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.meta_agent = None
        self.initialized = False
        
        if config_path and config_path != config.config_path:
            config.config_path = config_path
            config.load_config()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the reasoning engine."""
        try:
            agent_config = config.get_agent_config()
            agent_type = agent_config.get("type", "meta_agent")

            if agent_type == "meta_agent":
                self.meta_agent = MetaAgent()
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")

            self.initialized = True
            logger.info("Reasoning engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            self.initialized = False
    
    async def reason(self, question: str, schema: str, evidence: str, db_schema: DatabaseSchema, db_path: str, max_turns: Optional[int] = None, question_id: Optional[str] = None, retrieved_values: Optional[Dict[str, Dict[str, list]]] = None) -> Tuple[ReasoningSession, List[ReasoningStep]]:
        """Run inference"""
        if not self.initialized:
            raise RuntimeError("Reasoning engine not initialized")
        
        logger.debug("Starting reasoning session...")
        start_time = time.time()
        
        try:
            session, steps = await self.meta_agent.reason(question, schema, evidence, db_schema, db_path, question_id=question_id, retrieved_values=retrieved_values)
            
            execution_time = time.time() - start_time
            session.execution_time = execution_time
            logger.debug(f"Reasoning completed in {execution_time:.2f} seconds")
            
            return session, steps
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            error_session = ReasoningSession(
                question=question,
                schema=schema,
                db_id=db_schema.db_id if db_schema else "N/A",
                db_path=db_path,
                success=False
            )
            return error_session, []
    
    def update_worker_model(self, worker_type: WorkerType, new_port: int):
        """Update worker model ports"""
        if not self.initialized:
            raise RuntimeError("Reasoning engine not initialized")
        
        self.meta_agent.update_worker_model(worker_type, new_port)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        config_info = {
            "models": {name: {
                "base_url": model.base_url,
                "model_name": model.model_name,
                "temperature": model.temperature
            } for name, model in config.models.items()},
            "workers": {name: {
                "model_port": worker.model_port,
                "temperature": worker.temperature
            } for name, worker in config.workers.items()},
            "meta_agent": {
                "model_port": config.meta_agent_config.get("model_port", 9999),
                "max_turns": config.meta_agent_config.get("max_turns", 8)
            }
        }
        
        return {
            "status": "initialized",
            "config": config_info,
            "meta_agent_turns": self.meta_agent.max_turns if self.meta_agent else None
        }
    
    def save_session(self, session: ReasoningSession, steps: List[ReasoningStep], output_path: str):
        """Save inference sessions and traces"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        session_data = {
            "session_state": {
                "question": session.question,
                "schema": session.schema,
                "db_id": session.db_id,
                "evidence": session.evidence,
                "success": session.success,
                "final_sql": session.final_sql,
                "total_tokens": session.total_tokens,
                "execution_time": session.execution_time,
                "rephrased_question": session.rephrased_question,
                "selected_schema_context": session.selected_schema_context,
                "identified_column_values": session.identified_column_values,
                "identified_column_functions": session.identified_column_functions,
                "constructed_sql": session.constructed_sql,
                "refined_sql": session.refined_sql
            },
            "reasoning_trace": [
                {
                    "turn": step.turn,
                    "llm_response": step.llm_response,
                    "worker_call": step.worker_call,
                    "worker_result": {
                        "worker_type": {"value": step.worker_result.worker_type.value},
                        "success": step.worker_result.success,
                        "result": step.worker_result.result,
                        "error_message": step.worker_result.error_message,
                        "usage": step.worker_result.usage
                    } if step.worker_result else None,
                    "reasoning": step.reasoning
                }
                for step in steps
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Session and trace saved to {output_path}")

    def load_session(self, session_path: str) -> Optional[Tuple[ReasoningSession, List[ReasoningStep]]]:
        """Load inference sessions and traces"""
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            steps = []
            for step_data in session_data.get("reasoning_trace", []):
                worker_result_data = step_data.get("worker_result")
                worker_result = None
                if worker_result_data:
                    worker_type_str = worker_result_data.get("worker_type", {}).get("value")
                    worker_type_enum = next((wt for wt in WorkerType if wt.value == worker_type_str), None)

                    if worker_type_enum:
                        worker_result = WorkerResult(
                            worker_type=worker_type_enum,
                            result=worker_result_data.get("result"),
                            turn=step_data.get("turn"),
                            success=worker_result_data.get("success"),
                            error_message=worker_result_data.get("error_message"),
                            usage=worker_result_data.get("usage")
                        )
                
                step = ReasoningStep(
                    turn=step_data.get("turn"),
                    llm_response=step_data.get("llm_response"),
                    worker_call=step_data.get("worker_call"),
                    worker_result=worker_result,
                    reasoning=step_data.get("reasoning", "")
                )
                steps.append(step)

            state_data = session_data.get("session_state", {})
            session = ReasoningSession(
                question=state_data.get("question", ""),
                schema=state_data.get("schema", ""),
                db_id=state_data.get("db_id", ""),
                db_path=state_data.get("db_path", ""),
                evidence=state_data.get("evidence"),
                final_sql=state_data.get("final_sql"),
                success=state_data.get("success", False),
                total_tokens=state_data.get("total_tokens", 0),
                execution_time=state_data.get("execution_time", 0.0),
                rephrased_question=state_data.get("rephrased_question"),
                selected_schema_context=state_data.get("selected_schema_context"),
                identified_column_values=state_data.get("identified_column_values"),
                identified_column_functions=state_data.get("identified_column_functions"),
                constructed_sql=state_data.get("constructed_sql"),
                refined_sql=state_data.get("refined_sql")
            )
            
            logger.debug(f"Session and trace loaded from {session_path}")
            return session, steps
            
        except Exception as e:
            logger.error(f"Failed to load session from {session_path}: {e}")
            return None, None
    
    def get_reasoning_summary(self, session: ReasoningSession) -> Dict[str, Any]:
        """Get the inference summary"""
        if not self.initialized:
            return {"error": "Engine not initialized"}
        
        return self.meta_agent.get_reasoning_summary(session)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration"""
        validation_results = {
            "models": {},
            "workers": {},
            "meta_agent": {},
            "overall": True
        }
        
        for model_name, model_config in config.models.items():
            try:
                if not model_config.base_url.startswith(("http://", "https://")):
                    validation_results["models"][model_name] = {"valid": False, "error": "Invalid base_url"}
                    validation_results["overall"] = False
                else:
                    validation_results["models"][model_name] = {"valid": True}
            except Exception as e:
                validation_results["models"][model_name] = {"valid": False, "error": str(e)}
                validation_results["overall"] = False
        
        for worker_name, worker_config in config.workers.items():
            try:
                if worker_config.model_port <= 0 or worker_config.model_port > 65535:
                    validation_results["workers"][worker_name] = {"valid": False, "error": "Invalid port number"}
                    validation_results["overall"] = False
                else:
                    validation_results["workers"][worker_name] = {"valid": True}
            except Exception as e:
                validation_results["workers"][worker_name] = {"valid": False, "error": str(e)}
                validation_results["overall"] = False
        
        try:
            meta_config = config.get_meta_agent_config()
            if meta_config.get("max_turns", 0) <= 0:
                validation_results["meta_agent"] = {"valid": False, "error": "Invalid max_turns"}
                validation_results["overall"] = False
            else:
                validation_results["meta_agent"] = {"valid": True}
        except Exception as e:
            validation_results["meta_agent"] = {"valid": False, "error": str(e)}
            validation_results["overall"] = False
        
        return validation_results

reasoning_engine = None

def get_reasoning_engine(config_path: Optional[str] = None) -> ReasoningEngine:
    """Get the reasoning engine instance."""
    global reasoning_engine
    
    if reasoning_engine is None or (config_path and config_path != reasoning_engine.config_path):
        reasoning_engine = ReasoningEngine(config_path)
    
    return reasoning_engine
