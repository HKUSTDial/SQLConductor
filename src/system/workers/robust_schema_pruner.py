import re
import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from pathlib import Path
from loguru import logger

from src.system.types import WorkerType, WorkerInput, WorkerResult
from src.system.workers.base import BaseWorker
from src.database.schema import TableSchema, DatabaseSchema
from src.database.utils import build_table_profile_statement, map_lower_table_name_to_original_table_name, map_lower_column_name_to_original_column_name
from src.system.prompt_factory import get_worker_prompt, get_prompt, get_robust_icl_prompt
from src.system.workers.utils import _build_hint_from_steps, _extract_sql_query_answer, merge_schema_linking_results, get_icl_examples_path

class RobustSchemaPrunerWorker(BaseWorker):
    """
    Robust Schema Pruner Worker.
    Combines Direct Linking, Reversed Linking, and Value Linking.
    """
    
    def __init__(self, worker_type: WorkerType):
        super().__init__(worker_type)
        self.description = "Use to identify *only* the tables and columns relevant to the question. This reduces noise for the SQL generator. **Highly recommended as an early step.**"
        self.direct_linking_budget = 8
        self.reversed_linking_budget = 8
        self.value_distance_threshold = 0.02
        self.max_retry = 5
        self._icl_examples_cache = None
        self._icl_examples_path = get_icl_examples_path()

    def _get_icl_examples(self, question_id: str) -> List[Dict[str, str]]:
        """Load ICL examples from the config-derived path."""
        if self._icl_examples_cache is None:
            if self._icl_examples_path:
                try:
                    with open(self._icl_examples_path, "r") as f:
                        self._icl_examples_cache = json.load(f)
                        logger.trace(f"[Robust Schema Pruner] Loaded ICL examples from {self._icl_examples_path}")
                except Exception as e:
                    logger.error(f"[Robust Schema Pruner] Failed to load ICL examples: {e}")
                    self._icl_examples_cache = {}
            else:
                self._icl_examples_cache = {}

        return self._icl_examples_cache.get(str(question_id), [])

    def _format_icl_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples for the prompt."""
        if not examples:
            return ""
        
        formatted = ""
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Question: {ex.get('question', '')}\n"
            formatted += f"SQL: {ex.get('sql', '')}\n\n"
        return formatted.strip()
    
    async def execute(self, input_data: WorkerInput) -> WorkerResult:
        self.turn_count += 1
        
        try:
            table_schema_dict = input_data.context.get("table_schema_dict")
            if not table_schema_dict:
                raise ValueError("table_schema_dict not found in WorkerInput context.")
            
            db_schema = input_data.context.get("db_schema")
            
            schema_dict_for_mapping = {"tables": {k: v.to_dict() for k, v in table_schema_dict.items()}}
            
            value_result, value_usage = await self._value_linking(input_data, schema_dict_for_mapping)
            
            results = await asyncio.gather(
                self._direct_linking(input_data, schema_dict_for_mapping, table_schema_dict),
                self._reversed_linking(input_data, schema_dict_for_mapping, table_schema_dict)
            )
            
            direct_result, direct_usage = results[0]
            reversed_result, reversed_usage = results[1]
            
            if direct_result is None or reversed_result is None or value_result is None:
                failed_linkers = []
                if direct_result is None: failed_linkers.append("direct")
                if reversed_result is None: failed_linkers.append("reversed")
                if value_result is None: failed_linkers.append("value")
                
                logger.error(f"[Robust Schema Pruner] Linker(s) {failed_linkers} failed. Skipping pruning and retaining full schema.")
                
                return WorkerResult(
                    worker_type=self.worker_type,
                    result=input_data.schema,
                    turn=self.turn_count,
                    success=True,
                    usage={
                        "prompt_tokens": direct_usage.get("prompt_tokens", 0) + reversed_usage.get("prompt_tokens", 0) + value_usage.get("prompt_tokens", 0),
                        "completion_tokens": direct_usage.get("completion_tokens", 0) + reversed_usage.get("completion_tokens", 0) + value_usage.get("completion_tokens", 0),
                        "total_tokens": direct_usage.get("total_tokens", 0) + reversed_usage.get("total_tokens", 0) + value_usage.get("total_tokens", 0)
                    }
                )
            
            merged_selection = merge_schema_linking_results([direct_result, reversed_result, value_result])

            schema_context = self._generate_final_schema_context(merged_selection, table_schema_dict)
            
            total_usage = {
                "prompt_tokens": direct_usage.get("prompt_tokens", 0) + reversed_usage.get("prompt_tokens", 0) + value_usage.get("prompt_tokens", 0),
                "completion_tokens": direct_usage.get("completion_tokens", 0) + reversed_usage.get("completion_tokens", 0) + value_usage.get("completion_tokens", 0),
                "total_tokens": direct_usage.get("total_tokens", 0) + reversed_usage.get("total_tokens", 0) + value_usage.get("total_tokens", 0),
            }
            
            return WorkerResult(
                worker_type=self.worker_type,
                result=schema_context,
                turn=self.turn_count,
                success=True,
                usage=total_usage
            )
            
        except Exception as e:
            logger.error(f"[Robust Schema Pruner] Failed: {e}")
            return WorkerResult(
                worker_type=self.worker_type,
                result=input_data.schema,
                turn=self.turn_count,
                success=False,
                error_message=str(e)
            )

    async def _direct_linking(self, input_data: WorkerInput, schema_dict: Dict, table_schema_dict: Dict[str, TableSchema]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Direct Linking Strategy
        """
        try:
            database_schema_profile = self._get_database_schema_profile(table_schema_dict)
            steps = input_data.context.get("steps", [])
            hint = _build_hint_from_steps(steps, input_data.context)
            
            prompt = get_prompt(
                "schema_linking_direct",
                DATABASE_SCHEMA=database_schema_profile,
                QUESTION=input_data.question,
                HINT=hint or "None"
            ).strip()
            
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_selections = []
            
            remaining = self.direct_linking_budget
            retry_count = 0
            while remaining > 0 and len(all_selections) < self.direct_linking_budget and retry_count < self.max_retry:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.model_config.max_tokens,
                    n=remaining,
                    stop=self.config.model_config.stop
                )
                
                if response.usage:
                    total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += response.usage.get("total_tokens", 0)
                
                if response.error:
                    logger.warning(f"Direct linking LLM error: {response.error}")
                    break
                
                contents = response.contents or ([response.content] if response.content else [])
                for content in contents:
                    selection = self._parse_direct_linking_response(content, schema_dict)
                    if selection:
                        all_selections.append(selection)
                
                remaining = self.direct_linking_budget - len(all_selections)
                retry_count += 1
            
            if retry_count >= self.max_retry and len(all_selections) < self.direct_linking_budget:
                logger.warning(f"[Direct Linking] Reached max_retry ({self.max_retry}), got {len(all_selections)}/{self.direct_linking_budget} valid selections")
            logger.trace(f"[Direct Linking] Completed {len(all_selections)} selections.")
            return merge_schema_linking_results(all_selections), total_usage
            
        except Exception as e:
            logger.error(f"[Direct Linking] Failed: {e}")
            return None, {}

    async def _reversed_linking(self, input_data: WorkerInput, schema_dict: Dict, table_schema_dict: Dict[str, TableSchema]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Reversed Linking Strategy: Generate SQL using ICL -> Extract Schema
        """
        try:
            steps = input_data.context.get("steps", [])
            hint = _build_hint_from_steps(steps, input_data.context)
            
            question_id = input_data.context.get("question_id")
            icl_examples = (
                self._get_icl_examples(str(question_id))
                if input_data.context.get("use_icl_examples", True) and question_id is not None
                else []
            )
            
            prompt = get_robust_icl_prompt(
                FEW_SHOT_EXAMPLES=self._format_icl_examples(icl_examples),
                DATABASE_SCHEMA=input_data.schema,
                QUESTION=input_data.question,
                HINT=hint
            ).strip()
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            all_selections = []
            
            remaining = self.reversed_linking_budget
            retry_count = 0
            while remaining > 0 and len(all_selections) < self.reversed_linking_budget and retry_count < self.max_retry:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.model_config.max_tokens,
                    n=remaining,
                    stop=["</result>"]
                )
                
                if response.usage:
                    total_usage["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += response.usage.get("completion_tokens", 0)
                    total_usage["total_tokens"] += response.usage.get("total_tokens", 0)
                
                if response.error:
                    logger.warning(f"Reversed linking LLM error: {response.error}")
                    break

                contents = response.contents or ([response.content] if response.content else [])
                for content in contents:
                    # Restore the closing tag omitted by the stop sequence.
                    if "</result>" not in content:
                        content += "</result>"
                    
                    sql = _extract_sql_query_answer(content)
                    if not sql:
                        match = re.search(r"```sql\s*(.*?)\s*```", content, re.DOTALL)
                        if match:
                            sql = match.group(1).strip()
                    
                    if sql:
                        selection = self._extract_tables_and_columns_from_sql(sql, schema_dict)
                        if selection:
                            all_selections.append(selection)
                
                remaining = self.reversed_linking_budget - len(all_selections)
                retry_count += 1
            
            if retry_count >= self.max_retry and len(all_selections) < self.reversed_linking_budget:
                logger.warning(f"[Reversed Linking] Reached max_retry ({self.max_retry}), got {len(all_selections)}/{self.reversed_linking_budget} valid selections")
            
            logger.trace(f"[Reversed Linking] Done. Selections: {len(all_selections)}")
            return merge_schema_linking_results(all_selections), total_usage

        except Exception as e:
            logger.error(f"[Reversed Linking] Failed: {e}")
            return None, {}

    async def _value_linking(self, input_data: WorkerInput, schema_dict: Dict) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Value Linking Strategy
        """
        retrieved_values = input_data.context.get("retrieved_values")
        if not retrieved_values:
            logger.trace(f"[Value Linking] No retrieved values found in context.")
            return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
        linked_tables_and_columns = defaultdict(list)
        try:
            for table_name, columns in retrieved_values.items():
                original_table_name = map_lower_table_name_to_original_table_name(table_name, schema_dict)
                if original_table_name is None:
                    continue
                for column_name, values in columns.items():
                    # Values may be a list of dicts with 'distance' or a list of
                    # strings if the distance check was already done.

                    is_relevant = False
                    if isinstance(values, list):
                        if not values:
                            continue
                        first_val = values[0]
                        if isinstance(first_val, dict) and "distance" in first_val:
                            if any(v["distance"] < self.value_distance_threshold for v in values):
                                is_relevant = True
                        else:
                            is_relevant = True
                    
                    if is_relevant:
                        original_column_name = map_lower_column_name_to_original_column_name(original_table_name, column_name, schema_dict)
                        if original_column_name is None:
                            continue
                        linked_tables_and_columns[original_table_name].append(original_column_name)
            
            logger.trace(f"[Value Linking] Selected {len(linked_tables_and_columns)} tables.")
            return dict(linked_tables_and_columns), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e:
            logger.error(f"[Value Linking] Failed: {e}")
            return None, {}

    def _get_database_schema_profile(self, table_schema_dict: Dict[str, TableSchema]) -> str:
        """
        Generates the text representation of the schema (DDLs)
        """
        ddls = []
        for table_name, schema in table_schema_dict.items():
            schema_dict = schema.to_dict()
            ddls.append(build_table_profile_statement(schema_dict))
        return "\n\n".join(ddls)

    def _parse_direct_linking_response(self, response: str, schema_dict: Dict) -> Optional[Dict[str, List[str]]]:
        """
        Parses XML response from Direct Linking
        """
        try:
            answer_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if not answer_match:
                return None
            
            answer_content = answer_match.group(1).strip()
            result = {}
            
            table_matches = re.findall(r'<table\s+table_name="([^"]+)"[^>]*>(.*?)</table>', answer_content, re.DOTALL)
            
            for table_name, table_content in table_matches:
                original_table_name = map_lower_table_name_to_original_table_name(table_name, schema_dict)
                if original_table_name is None:
                    continue
                
                result[original_table_name] = []
                
                column_matches = re.findall(r'<column\s+column_name="([^"]+)"', table_content)
                
                for column_name in column_matches:
                    original_column_name = map_lower_column_name_to_original_column_name(original_table_name, column_name, schema_dict)
                    if original_column_name is None:
                        continue
                    result[original_table_name].append(original_column_name)
            
            return result if result else None
            
        except Exception as e:
            logger.warning(f"Error parsing direct linking response: {e}")
            return None

    def _extract_tables_and_columns_from_sql(self, sql: str, schema_dict: Dict) -> Dict[str, List[str]]:
        """
        Extracts tables and columns mentioned in SQL
        """
        
        all_table_names = list(schema_dict["tables"].keys())
        all_column_names = []
        for t in schema_dict["tables"].values():
            all_column_names.extend(t["columns"].keys())
            
        sql_lower = sql.lower()
        
        table_names = list(set([t.lower() for t in all_table_names if t.lower() in sql_lower]))
        column_names = list(set([c.lower() for c in all_column_names if c.lower() in sql_lower]))
        
        used_tables_and_columns = {}
        for table_name in table_names:
            original_table_name = map_lower_table_name_to_original_table_name(table_name, schema_dict)
            if original_table_name is None:
                continue
            used_tables_and_columns[original_table_name] = []
            for column_name in column_names:
                original_column_name = map_lower_column_name_to_original_column_name(original_table_name, column_name, schema_dict)
                if original_column_name is None:
                    continue
                used_tables_and_columns[original_table_name].append(original_column_name)
                
        return used_tables_and_columns

    def _generate_final_schema_context(self, selection: Dict[str, List[str]], table_schema_dict: Dict[str, TableSchema]) -> str:
        """
        Generates the final DDLs for the selected schema
        """
        final_selection = {}
        for table_name, columns in selection.items():
            if table_name in table_schema_dict:
                final_selection[table_name] = set(columns)
        
        for table_name in list(final_selection.keys()):
            table_schema = table_schema_dict[table_name]
            
            for col in table_schema.columns.values():
                if col.primary_key:
                    final_selection[table_name].add(col.original_column_name)
                    
            for col in table_schema.columns.values():
                if col.foreign_keys:
                    for target_table_name, target_column_name in col.foreign_keys:
                        # Only keep FK if target table is ALSO in the global selection
                        if target_table_name in final_selection:
                            final_selection[table_name].add(col.original_column_name)
                            final_selection[target_table_name].add(target_column_name)

        ddls = []
        for table_name, columns_to_keep in final_selection.items():
            original_table_schema = table_schema_dict[table_name]
            
            filtered_columns = {}
            for col_name, col in original_table_schema.columns.items():
                if col_name in columns_to_keep:
                    from copy import deepcopy
                    col_copy = deepcopy(col)
                    
                    # Remove FK references if the target table is NOT in the selection
                    valid_fks = []
                    for target_table_name, target_column_name in col_copy.foreign_keys:
                        if target_table_name in final_selection and target_column_name in final_selection[target_table_name]:
                            valid_fks.append((target_table_name, target_column_name))
                    col_copy.foreign_keys = valid_fks
                    
                    filtered_columns[col_name] = col_copy
            
            filtered_schema = TableSchema(
                table_name=original_table_schema.table_name,
                columns=filtered_columns
            )
            
            ddls.append(build_table_profile_statement(filtered_schema.to_dict()))
            
        return "\n\n".join(ddls)
