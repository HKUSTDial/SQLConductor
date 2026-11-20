import sqlite3
import re as _re
import threading
from enum import Enum
from typing import Optional, List, Tuple
from functools import lru_cache
from prettytable import PrettyTable
import sqlglot


def _sqlite_regexp(pattern: str, string: str) -> bool:
    """Implement SQLite REGEXP matching."""
    if string is None or pattern is None:
        return False
    try:
        return bool(_re.search(pattern, str(string)))
    except _re.error:
        return False


def _register_custom_functions(conn: sqlite3.Connection) -> None:
    """Register custom scalar functions on a SQLite connection."""
    conn.create_function("regexp", 2, _sqlite_regexp, deterministic=True)

class SQLExecutionResultType(Enum):
    """
    Type of the result of a SQL query execution.

    Attributes:
        SUCCESS: The query is executed successfully with non-empty, non-null results.
        TIMEOUT: The query execution timed out.
        ERROR: The query execution failed.
        EMPTY_RESULT: The query executed successfully but returned 0 rows.
        ALL_NULL_RESULT: The query executed successfully but all values are NULL.
    """
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    EMPTY_RESULT = "empty_result"
    ALL_NULL_RESULT = "all_null_result"
    
class SQLExecutionResult:
    """
    Result of a SQL query execution.
    """
    def __init__(self, db_path: str, sql: str, result_type: SQLExecutionResultType, result_cols: Optional[List[str]], result: Optional[List[Tuple]], error_message: Optional[str]) -> None:
        self.db_path = db_path
        self.sql = sql
        self.result_type = result_type
        self.result_cols = result_cols
        self.result = result
        self.error_message = error_message
        
    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "sql": self.sql,
            "result_type": self.result_type.value,
            "result_cols": self.result_cols,
            "result": self.result,
            "error_message": self.error_message
        }

class ExecuteSQLThread(threading.Thread):
    """
    Thread to execute a SQL query.
    """
    def __init__(self, db_path: str, query: str, timeout: int) -> None:
        super().__init__()
        self.db_path = db_path
        self.query = query
        self.timeout = timeout
        self.result_cols = None
        self.result = None
        self.exception = None
        
        self.stop_event = threading.Event()
        
    def run(self) -> None:
        def check_stop():
            if self.stop_event.is_set():
                raise Exception("Query execution cancelled")
        
        try:
            # Keep query execution read-only.
            with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as conn:
                _register_custom_functions(conn)
                conn.text_factory = lambda x: str(x, 'utf-8', errors='replace')
                conn.set_progress_handler(check_stop, 1000)
                cursor = conn.cursor()
                cursor.execute(self.query)
                self.result_cols = [col[0] for col in cursor.description]
                self.result = cursor.fetchall()
        except Exception as e:
            self.exception = e

def execute_sql_with_timeout(db_path: str, query: str, timeout: int = 60) -> SQLExecutionResult:
    """
    Execute a SQL query synchronously with a timeout.
    
    Args:
        db_path: The path to the database.
        query: The SQL query to execute.
        timeout: The timeout.
    Returns:
        The result of the SQL query.
    """ 
    thread = ExecuteSQLThread(db_path, query, timeout)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        thread.stop_event.set()
        thread.join(1)
        error_message = f"SQL execution timed out after {timeout} seconds"
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.TIMEOUT, None, None, error_message)
    if thread.exception:
        error_message = str(thread.exception)
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.ERROR, None, None, error_message)
    if thread.result is not None and len(thread.result) == 0:
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.EMPTY_RESULT, thread.result_cols, thread.result,
                                  "The SQL query returned an empty result table.")
    if thread.result is not None and not any(any(val is not None for val in row) for row in thread.result):
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.ALL_NULL_RESULT, thread.result_cols, thread.result,
                                  "The SQL query returned a result table with all null values.")
    return SQLExecutionResult(db_path, query, SQLExecutionResultType.SUCCESS, thread.result_cols, thread.result, None)

def execute_sql_without_timeout(db_path: str, query: str) -> SQLExecutionResult:
    """
    Execute a SQL query without a timeout.
    
    Args:
        db_path: The path to the database.
        query: The SQL query to execute.
    Returns:
        The result of the SQL query.
    """
    try:
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            _register_custom_functions(conn)
            conn.text_factory = lambda x: str(x, 'utf-8', errors='replace')
            cursor = conn.cursor()
            cursor.execute(query)
            result_cols = [col[0] for col in cursor.description]
            result = cursor.fetchall()
            if len(result) == 0:
                return SQLExecutionResult(db_path, query, SQLExecutionResultType.EMPTY_RESULT, result_cols, result,
                                          "The SQL query returned an empty result table.")
            if not any(any(val is not None for val in row) for row in result):
                return SQLExecutionResult(db_path, query, SQLExecutionResultType.ALL_NULL_RESULT, result_cols, result,
                                          "The SQL query returned a result table with all null values.")
            return SQLExecutionResult(db_path, query, SQLExecutionResultType.SUCCESS, result_cols, result, None)
    except Exception as e:
        return SQLExecutionResult(db_path, query, SQLExecutionResultType.ERROR, None, None, str(e))

def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL query.
    
    Args:
        sql: The SQL query to normalize.
    Returns:
        The normalized SQL query.
    """
    sql = sql.strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3]
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        return parsed.sql(dialect="sqlite", normalize=True, pretty=False, comments=False)
    except Exception as e:
        return sql

@lru_cache(maxsize=10000)
def _cached_execute_sql_with_timeout(db_path: str, sql_query: str, timeout: int = 60) -> SQLExecutionResult:
    result = execute_sql_with_timeout(db_path, sql_query, timeout)
    return result

def cached_execute_sql_with_timeout(db_path: str, sql_query: str, timeout: int = 60) -> SQLExecutionResult:
    result = _cached_execute_sql_with_timeout(db_path, sql_query, timeout)
    return result

def is_valid_execution_result(result: SQLExecutionResult) -> bool:
    """Returns True if the SQL executed without errors (SUCCESS, EMPTY_RESULT, or ALL_NULL_RESULT)."""
    return result.result_type in (
        SQLExecutionResultType.SUCCESS,
        SQLExecutionResultType.EMPTY_RESULT,
        SQLExecutionResultType.ALL_NULL_RESULT,
    )

def has_non_empty_result(result: SQLExecutionResult) -> bool:
    """Returns True only if the SQL executed successfully with non-empty, non-null results."""
    return result.result_type == SQLExecutionResultType.SUCCESS

def format_execution_result(result: SQLExecutionResult, row_limit: int = 3, val_length_limit: int = 100) -> str:
    if is_valid_execution_result(result) and result.result_cols is not None:
        table = PrettyTable()
        # PrettyTable requires unique column names.
        cols = list(result.result_cols)
        if len(cols) != len(set(cols)):
            cols = [col + "_" + str(i) for i, col in enumerate(cols)]
        table.field_names = cols
        rows = result.result or []
        truncated_result = []
        for row in rows[:row_limit]:
            truncated_row = []
            for i, val in enumerate(row):
                if isinstance(val, str) and len(val) > val_length_limit:
                    truncated_row.append(val[:val_length_limit] + "...")
                else:
                    truncated_row.append(val)
            truncated_result.append(truncated_row)
        table.add_rows(truncated_result)
        return str(table)
    else:
        return result.error_message or "Unknown error"
