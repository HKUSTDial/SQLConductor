import re
import json
import copy
import random
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.system.types import WorkerType, WorkerInput, WorkerResult
from src.system.prompt_factory import get_worker_prompt
from src.system.workers.base import BaseWorker
from src.system.workers.utils import (
    _build_hint_from_steps, 
    SCHEMA_PRUNER_LLM_KWARGS_N
)
from src.database.schema import TableSchema
from src.database.utils import build_table_profile_statement

class SchemaPrunerWorker(BaseWorker):
    """Worker that prunes the schema."""

    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "Use to identify *only* the tables and columns relevant to the question. This reduces noise for the SQL generator. **Highly recommended as an early step.**"
    
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        self.turn_count += 1
        
        try:
            prompt = self._build_prompt(input_data)
            table_schema_dict = input_data.context.get("table_schema_dict")
            if not table_schema_dict:
                raise ValueError("table_schema_dict not found in WorkerInput context for SchemaPrunerWorker.")
            
            schema_context, usage = await self._generate_schema_context(prompt, table_schema_dict, input_data.schema)
            
            return WorkerResult(
                worker_type=self.worker_type,
                result=schema_context,
                turn=self.turn_count,
                success=True,
                usage=usage
            )
        except Exception as e:
            logger.error(f"Worker {self.worker_type.value} execution failed: {e}")
            return WorkerResult(
                worker_type=self.worker_type,
                result=input_data.schema,
                turn=self.turn_count,
                success=False,
                error_message=str(e)
            )
    
    def _build_prompt(self, input_data: WorkerInput) -> str:
        steps = input_data.context.get("steps", [])
        hint = _build_hint_from_steps(steps, input_data.context)
        
        return get_worker_prompt(
            self.worker_type,
            QUESTION=input_data.question,
            HINT=hint,
            SCHEMA_CONTEXT=input_data.schema
        )

    async def _generate_schema_context(
        self,
        prompt: str,
        table_schema_dict: Dict[str, TableSchema],
        fallback_schema: str
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        unique_selection_hashes = set()
        sampled_schema_dicts: List[Dict[str, TableSchema]] = []
        
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.model_config.max_tokens,
            n=SCHEMA_PRUNER_LLM_KWARGS_N,
            stop=self.config.model_config.stop
        )
        
        if response.error:
            logger.error(f"LLM call failed: {response.error}")
        else:
            response_contents = response.contents or ([response.content] if response.content else [])
            for idx, content in enumerate(response_contents, 1):
                logger.trace(
                    f"[{self.worker_type.value}] Received schema selection sample {idx}."
                )
            for content in response_contents:
                new_table_schema_dict, selection_dict = self.select_schema(table_schema_dict, content)
                if not new_table_schema_dict or not selection_dict:
                    continue
                
                selection_hash = self._hash_selection_dict(selection_dict)
                if selection_hash in unique_selection_hashes:
                    continue
                
                unique_selection_hashes.add(selection_hash)
                sampled_schema_dicts.append(new_table_schema_dict)
        
        if not sampled_schema_dicts:
            logger.warning("Schema selection failed after multiple attempts. Returning the original schema.")
            return fallback_schema, response.usage
        
        merged_schema_dict = self._merge_schema_dicts(sampled_schema_dicts)
        schema_context_ddl_list = [build_table_profile_statement(
            table_schema.to_dict(), 
            add_value_description=True,
            add_column_description=True,
            add_value_examples=True,
            add_expanded_column_name=True
        ) for table_schema in merged_schema_dict.values()]
        random.shuffle(schema_context_ddl_list)
        schema_context = "\n".join(schema_context_ddl_list)
        return schema_context, response.usage
    
    @staticmethod
    def _hash_selection_dict(selection_dict: Dict[str, List[str]]) -> frozenset:
        return frozenset(
            (table_name.lower(), column_name.lower())
            for table_name, column_names in selection_dict.items()
            for column_name in column_names
        )
    
    @staticmethod
    def _merge_schema_dicts(schema_dicts: List[Dict[str, TableSchema]]) -> Dict[str, TableSchema]:
        merged: Dict[str, TableSchema] = {}
        for schema_dict in schema_dicts:
            for table_name, table_schema in schema_dict.items():
                if table_name not in merged:
                    merged[table_name] = copy.deepcopy(table_schema)
                else:
                    existing_table = merged[table_name]
                    for column_name, column_schema in table_schema.columns.items():
                        if column_name not in existing_table.columns:
                            existing_table.columns[column_name] = copy.deepcopy(column_schema)
        return merged

    def select_schema(self, table_schema_dict: Dict[str, TableSchema], schema_selection_response: str) -> Tuple[Optional[Dict[str, TableSchema]], Optional[Dict[str, List[str]]]]:
        try:
            schema_selection_dict = json.loads(re.search(r"```json\n(.*)```", schema_selection_response, flags=re.DOTALL).group(1))
        except Exception as e:
            logger.error(f"Error parsing schema selection response: {e}")
            return None, None

        selected_table_names_lower = [selected_table_name.lower() for selected_table_name in schema_selection_dict.keys()]
        
        new_table_schema_dict = {}
        for selected_table_name, selected_column_names in schema_selection_dict.items():
            for original_table_name, original_table_schema in table_schema_dict.items():
                if selected_table_name.lower() == original_table_name.lower():
                    new_table_schema = TableSchema(table_name=original_table_name, columns={})
                    for original_column_name, original_column_schema in original_table_schema.columns.items():
                        is_selected = False
                        for selected_column_name in selected_column_names:
                            if selected_column_name.lower() == original_column_name.lower():
                                is_selected = True
                                break
                        if original_column_schema.primary_key:
                            # Primary key columns must always be selected.
                            is_selected = True
                        for target_table_name, target_column_name in original_column_schema.foreign_keys:
                            if target_table_name.lower() in selected_table_names_lower:
                                # Foreign key columns must be selected when the target table is selected.
                                is_selected = True
                                break
                        for source_table_name, source_column_name in original_column_schema.referenced_by:
                            if source_table_name.lower() in selected_table_names_lower:
                                # Referenced-by columns must be selected when the source table is selected.
                                is_selected = True
                                break
                        if is_selected:
                            new_column_schema = copy.deepcopy(original_column_schema)
                            _foreign_keys = []
                            _referenced_by = []
                            for target_table_name, target_column_name in original_column_schema.foreign_keys:
                                if target_table_name.lower() in selected_table_names_lower:
                                    _foreign_keys.append((target_table_name, target_column_name))
                            for source_table_name, source_column_name in original_column_schema.referenced_by:
                                if source_table_name.lower() in selected_table_names_lower:
                                    _referenced_by.append((source_table_name, source_column_name))
                            new_column_schema.foreign_keys = _foreign_keys
                            new_column_schema.referenced_by = _referenced_by
                            new_table_schema.columns[original_column_name] = new_column_schema
                    if new_table_schema.columns:
                        new_table_schema_dict[original_table_name] = new_table_schema
        
        return new_table_schema_dict, schema_selection_dict
