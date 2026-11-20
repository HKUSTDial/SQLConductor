#!/usr/bin/env python3
"""Load worker prompt templates and format their inputs."""

from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from src.system.types import (
    WorkerType, ACTION_WORKER_MAP, WORKER_TEMPLATE_MAP
)
from src.system.config import config


TEMPLATE_DIR = Path("src/templates")

TEMPLATE_CACHE: Dict[str, str] = {}

PROMPT_CACHE = {}


def get_prompt(template_name: str, **kwargs) -> str:
    """Load and format a prompt template."""
    if template_name not in PROMPT_CACHE:
        template_dir = Path(config.config_data.get('system', {}).get('templates_dir', 'src/templates'))
        template_path = template_dir / f"{template_name}.txt"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            PROMPT_CACHE[template_name] = f.read()
            
    template = PROMPT_CACHE[template_name]
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        if not kwargs:
            return template
        logger.error(f"Missing template parameter for '{template_name}': {e}")
        raise ValueError(f"Template parameter missing: {e}")


def get_worker_type_for_action(action_name: str) -> Optional[WorkerType]:
    """Return the worker type for an action name."""
    for action_enum, worker_type in ACTION_WORKER_MAP.items():
        if action_enum.value == action_name:
            return worker_type
    return None


def get_action_name_for_worker(worker_type: WorkerType) -> Optional[str]:
    """Return the action name for a worker type."""
    for action_enum, worker_enum in ACTION_WORKER_MAP.items():
        if worker_enum == worker_type:
            return action_enum.value
    return None


def get_template_name_for_worker(worker_type: WorkerType) -> Optional[str]:
    """Return the template name for a worker type."""
    return WORKER_TEMPLATE_MAP.get(worker_type)


def load_template(template_name: str) -> str:
    """Load the contents of a template file."""
    if template_name in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[template_name]
    
    template_file = TEMPLATE_DIR / f"{template_name}.txt"
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    try:
        with open(template_file, "r", encoding="utf-8") as f:
            template_content = f.read()
            TEMPLATE_CACHE[template_name] = template_content
            return template_content
    except Exception as e:
        logger.error(f"Failed to load template {template_name}: {e}")
        raise


def get_worker_prompt(worker_type: WorkerType, **kwargs) -> str:
    """Build a worker prompt."""
    template_name = WORKER_TEMPLATE_MAP.get(worker_type)
    if not template_name:
        raise ValueError(f"No template found for worker type: {worker_type}")
    
    return get_prompt(
        template_name,
        **kwargs
    )


def get_meta_agent_system_prompt(worker_descriptions: Optional[str] = None) -> str:
    """Build the meta-agent system prompt."""
    return get_prompt("meta_agent_system", WORKER_DESCRIPTIONS=worker_descriptions or "")


def get_meta_agent_user_prompt(
    question: str, 
    schema: str,
    rephrased_question: Optional[str] = None,
    selected_schema_context: Optional[str] = None,
    identified_column_values: Optional[str] = None,
    identified_column_functions: Optional[str] = None,
    intermediate_sql_result: Optional[str] = None,
    history: Optional[str] = None,
    valid_next_workers_block: Optional[str] = None
) -> str:
    """Build the dynamic meta-agent user prompt."""
    
    context_parts = {}

    if rephrased_question:
        context_parts["Question"] = rephrased_question
    else:
        context_parts["Question"] = question

    if selected_schema_context:
        context_parts["Database Schema"] = f"This is the focused database schema after schema linking, based on the user's question.\n{selected_schema_context}"
    else:
        context_parts["Database Schema"] = schema

    if identified_column_values:
        context_parts["Identified Relevant Column Values"] = identified_column_values
    if identified_column_functions:
        context_parts["Identified Relevant Column Functions"] = identified_column_functions
    if intermediate_sql_result:
        context_parts["Last Generated SQL (with execution result)"] = intermediate_sql_result

    formatted_context = "\n".join([f"## {key}\n{value}" for key, value in context_parts.items()])
    
    return get_prompt(
        "meta_agent_user",
        CONTEXT_BLOCK=formatted_context,
        VALID_NEXT_WORKERS_BLOCK=valid_next_workers_block or "",
        HISTORY_BLOCK=history or "No history yet."
    )


def get_meta_agent_user_prompt_unrestricted(
    question: str,
    schema: str,
    rephrased_question: Optional[str] = None,
    selected_schema_context: Optional[str] = None,
    identified_column_values: Optional[str] = None,
    identified_column_functions: Optional[str] = None,
    intermediate_sql_result: Optional[str] = None,
    history: Optional[str] = None,
    valid_next_workers_block: Optional[str] = None
) -> str:
    """Build the unrestricted meta-agent user prompt using a dedicated template."""
    context_parts = {}
    if rephrased_question:
        context_parts["Question"] = rephrased_question
    else:
        context_parts["Question"] = question
    if selected_schema_context:
        context_parts["Database Schema"] = f"This is the focused database schema after schema linking, based on the user's question.\n{selected_schema_context}"
    else:
        context_parts["Database Schema"] = schema
    if identified_column_values:
        context_parts["Identified Relevant Column Values"] = identified_column_values
    if identified_column_functions:
        context_parts["Identified Relevant Column Functions"] = identified_column_functions
    if intermediate_sql_result:
        context_parts["Last Generated SQL (with execution result)"] = intermediate_sql_result

    formatted_context = "\n".join([f"## {key}\n{value}" for key, value in context_parts.items()])
    return get_prompt(
        "meta_agent_user_unrestricted",
        CONTEXT_BLOCK=formatted_context,
        VALID_NEXT_WORKERS_BLOCK=valid_next_workers_block or "",
        HISTORY_BLOCK=history or "No history yet."
    )


def list_available_templates() -> Dict[str, str]:
    """List all available templates"""
    templates = {}
    for template_file in TEMPLATE_DIR.glob("*.txt"):
        templates[template_file.stem] = str(template_file)
    return templates


def validate_template_parameters(template_name: str, **kwargs) -> bool:
    """Validate that all template parameters are provided."""
    try:
        template = load_template(template_name)
        template.format(**kwargs)
        return True
    except KeyError:
        return False


def get_template_parameters(template_name: str) -> set:
    """Get the parameter names required by the template."""
    try:
        template = load_template(template_name)
        import re
        params = re.findall(r'\{(\w+)\}', template)
        return set(params)
    except Exception:
        return set()


def reload_templates():
    """Clear cached templates for reloading on next use."""
    global TEMPLATE_CACHE
    TEMPLATE_CACHE.clear()
    logger.info("Template cache cleared")


def get_question_decomposer_prompt(**kwargs) -> str:
    """Get the question decomposer worker prompt."""
    return get_worker_prompt(WorkerType.QUESTION_DECOMPOSER, **kwargs)


def get_schema_pruner_prompt(**kwargs) -> str:
    """Get the schema selection worker prompt"""
    return get_worker_prompt(WorkerType.SCHEMA_PRUNER, **kwargs)


def get_value_extractor_prompt(**kwargs) -> str:
    """Get the value extraction worker prompt"""
    return get_worker_prompt(WorkerType.VALUE_EXTRACTOR, **kwargs)


def get_operation_identifier_prompt(**kwargs) -> str:
    """Get the operation identification worker prompt"""
    return get_worker_prompt(WorkerType.OPERATION_IDENTIFIER, **kwargs)


def get_sql_constructor_prompt(**kwargs) -> str:
    """Get the SQL construction worker prompt"""
    return get_worker_prompt(WorkerType.SQL_CONSTRUCTOR, **kwargs)


def get_sql_refiner_prompt(**kwargs) -> str:
    """Get the SQL refinement worker prompt"""
    return get_worker_prompt(WorkerType.SQL_REFINER, **kwargs)


def get_robust_dc_prompt(**kwargs) -> str:
    """Get prompt for Robust DC SQL Generation"""
    return get_prompt("robust_dc_sql_generation", **kwargs)


def get_robust_skeleton_prompt(**kwargs) -> str:
    """Get prompt for Robust Skeleton SQL Generation"""
    return get_prompt("robust_skeleton_sql_generation", **kwargs)


def get_robust_icl_prompt(**kwargs) -> str:
    """Get prompt for Robust ICL SQL Generation"""
    return get_prompt("robust_icl_sql_generation", **kwargs)


def get_br_pair_selection_prompt(**kwargs) -> str:
    """Get prompt for BR Pair Selection"""
    return get_prompt("br_pair_selection", **kwargs)


def get_robust_identify_column_values_prompt(**kwargs) -> str:
    """Get prompt for Robust Identify Column Values"""
    return get_prompt("robust_identify_column_values", **kwargs)


def get_robust_identify_column_functions_prompt(**kwargs) -> str:
    """Get prompt for Robust Identify Column Functions"""
    return get_prompt("robust_identify_column_functions", **kwargs)
