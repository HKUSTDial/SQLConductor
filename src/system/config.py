"""
Reasoning system configuration.
Supports flexible model configuration and port management.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
from loguru import logger
import sys

@dataclass
class ModelConfig:
    """Model configuration."""
    base_url: str
    api_key: str
    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: int = 2048
    timeout: int = 60
    stop: Optional[List[str]] = None

@dataclass
class WorkerConfig:
    """Worker configuration."""
    worker_type: str
    model_port: int
    model_config: Optional[ModelConfig] = None
    custom_prompt_template: Optional[str] = None
    temperature: float = 0.1
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class SystemConfig:
    log_level: str
    async_workers: bool
    result_cache: bool
    cache_ttl: int
    schema_max_length: int

@dataclass
class AgentConfig:
    type: str

@dataclass
class DataConfig:
    real_tasks_file: str

class ReasoningSystemConfig:
    """Configuration manager for the reasoning system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/exp/exp_config_default.yaml"
        self.worker_impl_config_path = "config/system/worker_impl_config.yaml"
        self.config_data = {}
        self.worker_impl_config = {}
        self.models = {}
        self.workers = {}
        self.load_config()
        self.load_worker_impl_config()
    
    def setup_logging(self):
        """Sets up the logger based on the configuration."""
        system_config = self.get_system_config()
        log_level = system_config.get("log_level", "DEBUG").upper()
        
        logger.remove()
        
        def format_log(record):
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            if record["level"].no >= logger.level("WARNING").no:
                if "question_id" in record["extra"]:
                    format_string += " | <yellow>[QID: {extra[question_id]}]</yellow>"
                if "db_id" in record["extra"]:
                    format_string += " <yellow>[DB: {extra[db_id]}]</yellow>"
            return format_string + "\n"

        logger.add(sys.stderr, format=format_log, level=log_level)
        logger.info(f"Logger configured with level: {log_level}")

    def load_config(self, config_path: Optional[str] = None):
        """Load a configuration file."""
        if config_path:
            self.config_path = config_path
            
        config_file = Path(self.config_path)
        
        if config_file.exists():
            if config_file.suffix == '.yaml':
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config_data = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        else:
            logger.warning(f"No config file found, using default configuration: {self.config_path}")
            self.config_data = self._get_default_config()
            self._save_default_config()
        
        self._parse_config()
    
    def load_worker_impl_config(self):
        """Load the worker implementation configuration."""
        path = Path(self.worker_impl_config_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.worker_impl_config = yaml.safe_load(f) or {}
                # Logger may not be configured yet when called from __init__.
            except Exception as e:
                logger.error(f"Failed to load worker implementation config: {e}")
                self.worker_impl_config = {}
        else:
            logger.warning(f"Worker implementation config not found at {path}, using defaults")
            self.worker_impl_config = {}

    def get_worker_impl(self, worker_type: str) -> str:
        """Return the implementation name used by a worker."""
        return self.worker_impl_config.get(worker_type, "default")
    
    def set_worker_impl(self, worker_type: str, impl_name: str):
        """Set the worker implementation name, usually from CLI arguments."""
        self.worker_impl_config[worker_type] = impl_name
        logger.info(f"Worker '{worker_type}' implementation set to: '{impl_name}'")
    
    def set_worker_impls_from_dict(self, worker_impls: Dict[str, str]):
        """Set worker implementation names in bulk."""
        for worker_type, impl_name in worker_impls.items():
            self.set_worker_impl(worker_type, impl_name)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration."""
        return {
            "models": {
                "default": {
                    "base_url": "http://localhost:9999/v1",
                    "api_key": "sk-demo",
                    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "max_tokens": 4096,
                    "timeout": 30
                }
            },
            "workers": {
                "Schema Pruner": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                },
                "Value Extractor": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                },
                "Operation Identifier": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                },
                "SQL Constructor": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                },
                "SQL Refiner": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                },
                "Question Decomposer": {
                    "model_port": 9999,
                    "temperature": 0.8,
                    "max_retries": 3
                }
            },
            "meta_agent": {
                "model_port": 9999,
                "temperature": 0.8,
                "max_tokens": 4096,
                "max_turns": 8
            },
            "system": {
                "log_level": "INFO",
                "async_workers": True,
                "result_cache": True,
                "cache_ttl": 3600,
                "schema_max_length": 24000
            },
            "data": {
                "real_tasks_file": "data/preprocessed/bird_task/dev/sds_tasks.pkl",
                "output_dir": "results/inference/test/demo"
            }
        }
    
    def _save_default_config(self):
        """Save the default configuration."""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Default configuration file saved to: {self.config_path}")
    
    def _parse_config(self):
        """Parse configuration data."""
        try:
            for model_name, model_data in self.config_data.get("models", {}).items():
                self.models[model_name] = ModelConfig(**model_data)
            
            for worker_name, worker_data in self.config_data.get("workers", {}).items():
                model_port = int(worker_data.get("model_port", 9999))
                model_name = self._get_model_name_by_port(model_port)
                
                worker_config = WorkerConfig(
                    worker_type=worker_name,
                    model_port=model_port,
                    model_config=self.models.get(model_name),
                    **{k: v for k, v in worker_data.items() if k not in ["model_port"]}
                )
                self.workers[worker_name] = worker_config
            
            meta_agent_data = self.config_data.get("meta_agent", {})
            model_port = int(meta_agent_data.get("model_port", 9999))
            model_name = self._get_model_name_by_port(model_port)
            
            self.meta_agent_config = {
                "model_config": self.models.get(model_name),
                **meta_agent_data
            }
            
            system_data = self.config_data.get("system", {})
            self.system_config = SystemConfig(
                log_level=system_data.get("log_level", "DEBUG"),
                async_workers=system_data.get("async_workers", True),
                result_cache=system_data.get("result_cache", True),
                cache_ttl=system_data.get("cache_ttl", 3600),
                schema_max_length=system_data.get("schema_max_length", 24000)
            )

            if 'agent' in system_data:
                self.agent_config = self._parse_agent_config(system_data['agent'])
            else:
                self.agent_config = AgentConfig(type="meta_agent")

            if 'data' in system_data:
                self.data_config = self._parse_data_config(system_data['data'])

        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            self._use_default_config()
    
    def _use_default_config(self):
        """Use default configuration"""
        logger.info("Using default configuration due to parsing error")
        
        self.models = {
            "default": ModelConfig(
                base_url="http://localhost:9999/v1",
                api_key="sk-demo",
                model_name="qwen2.5-coder-3b-instruct",
                temperature=0.0,
                top_p=1.0,
                max_tokens=2048,
                timeout=60
            )
        }
        
        self.workers = {}
        for worker_name in ["Schema Pruner", "Value Extractor", "Operation Identifier", 
                           "SQL Constructor", "SQL Refiner", "Question Decomposer"]:
            self.workers[worker_name] = WorkerConfig(
                worker_type=worker_name,
                model_port=9999,
                model_config=self.models["default"],
                temperature=0.1,
                max_retries=3,
                retry_delay=1.0
            )
        
        self.meta_agent_config = {
            "model_config": self.models["default"],
            "model_port": 9999,
            "temperature": 0.1,
            "max_tokens": 2048,
            "max_turns": 8
        }

        self.system_config = SystemConfig(
            log_level="INFO",
            async_workers=True,
            result_cache=True,
            cache_ttl=3600,
            schema_max_length=24000
        )

        self.agent_config = AgentConfig(type="meta_agent")

        self.data_config = DataConfig(real_tasks_file="tasks/real_tasks.json")
    
    def _get_model_name_by_port(self, port: int) -> str:
        """Get the model name by port"""
        port_str = str(port)
        for model_name, model_config in self.models.items():
            if port_str in model_config.base_url:
                return model_name
        return "default"
    
    def get_worker_config(self, worker_type: str) -> Optional[WorkerConfig]:
        """Get worker configuration"""
        return self.workers.get(worker_type)
    
    def get_meta_agent_config(self) -> Dict[str, Any]:
        """Get MetaAgent configuration"""
        return self.meta_agent_config
    
    def set_agent_type(self, agent_type: str):
        """Sets the agent type, typically from a command-line argument."""
        if self.agent_config:
            self.agent_config.type = agent_type
        else:
            self.agent_config = AgentConfig(type=agent_type)
        logger.info(f"Agent type set to: '{agent_type}'")

    def get_agent_config(self) -> Dict[str, Any]:
        """Get the agent configuration dictionary"""
        if not self.agent_config:
            return {"type": "meta_agent"}
        return {"type": self.agent_config.type}

    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.config_data.get("system", {})

    @property
    def unrestricted(self) -> bool:
        """Whether unrestricted mode is enabled (relaxes orchestration constraints)."""
        return self.config_data.get('system', {}).get('unrestricted', False)

    def save_config(self):
        """Save configuration to a file"""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)

    def _parse_agent_config(self, config_data: Dict[str, Any]) -> AgentConfig:
        """Parse agent configuration"""
        return AgentConfig(
            type=config_data.get('type', 'meta_agent')
        )
        
    def _parse_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Parse system configuration."""
        return SystemConfig(
            log_level=config_data.get("log_level", "INFO"),
            async_workers=config_data.get("async_workers", True),
            result_cache=config_data.get("result_cache", True),
            cache_ttl=config_data.get("cache_ttl", 3600),
            schema_max_length=config_data.get("schema_max_length", 24000)
        )

    def _parse_data_config(self, config_data: Dict[str, Any]) -> DataConfig:
        """Parse data configuration"""
        return DataConfig(
            real_tasks_file=config_data.get("real_tasks_file", "tasks/real_tasks.json")
        )

config = ReasoningSystemConfig()
