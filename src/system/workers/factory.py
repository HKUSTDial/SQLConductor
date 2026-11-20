from typing import Dict, Type
from loguru import logger
from src.system.types import WorkerType
from src.system.config import config
from src.system.workers.base import BaseWorker
from src.system.workers.schema_pruner import SchemaPrunerWorker
from src.system.workers.robust_schema_pruner import RobustSchemaPrunerWorker
from src.system.workers.value_extractor import ValueExtractorWorker
from src.system.workers.robust_value_extractor import RobustValueExtractorWorker
from src.system.workers.operation_identifier import OperationIdentifierWorker
from src.system.workers.robust_operation_identifier import RobustOperationIdentifierWorker
from src.system.workers.sql_constructor import SQLConstructorWorker
from src.system.workers.robust_sql_constructor import RobustSQLConstructorWorker
from src.system.workers.icl_sql_constructor import ICLSQLConstructorWorker
from src.system.workers.sql_refiner import SQLRefinerWorker
from src.system.workers.robust_sql_refiner import RobustSQLRefinerWorker
from src.system.workers.question_decomposer import QuestionDecomposerWorker
from src.system.workers.end import EndWorker

# Map WorkerType -> implementation name -> worker class.
WORKER_REGISTRY: Dict[WorkerType, Dict[str, Type[BaseWorker]]] = {
    WorkerType.QUESTION_DECOMPOSER: {
        "default": QuestionDecomposerWorker
    },
    WorkerType.SCHEMA_PRUNER: {
        "default": SchemaPrunerWorker,
        "robust": RobustSchemaPrunerWorker
    },
    WorkerType.VALUE_EXTRACTOR: {
        "default": ValueExtractorWorker,
        "robust": RobustValueExtractorWorker
    },
    WorkerType.OPERATION_IDENTIFIER: {
        "default": OperationIdentifierWorker,
        "robust": RobustOperationIdentifierWorker
    },
    WorkerType.SQL_CONSTRUCTOR: {
        "default": SQLConstructorWorker,
        "robust": RobustSQLConstructorWorker,
        "icl": ICLSQLConstructorWorker
    },
    WorkerType.SQL_REFINER: {
        "default": SQLRefinerWorker,
        "robust": RobustSQLRefinerWorker
    },
    WorkerType.END: {
        "default": EndWorker
    }
}

class WorkerFactory:
    """Worker factory."""
    
    @staticmethod
    def create_worker(worker_type: WorkerType, impl_name: str = None) -> BaseWorker:
        """Create a worker instance."""
        if impl_name is None:
            impl_name = config.get_worker_impl(worker_type.value)
        
        if worker_type not in WORKER_REGISTRY:
            raise ValueError(f"Unknown worker type: {worker_type}")
             
        implementations = WORKER_REGISTRY[worker_type]
        
        if impl_name not in implementations:
            raise ValueError(f"Unknown implementation '{impl_name}' for worker type {worker_type.value}. Available: {list(implementations.keys())}")
            
        worker_class = implementations[impl_name]
        logger.debug(f"'{worker_type.value}' Implementation: '{impl_name}'")
        return worker_class(worker_type)
