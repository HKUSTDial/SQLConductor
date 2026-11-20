"""
Prompt templates for SQL checkers
"""

EXECUTION_CHECKER_PROMPT = """
# Task:
You are an SQL database expert tasked with correcting a SQL query. A previous attempt to run a query did not yield the correct results, either due to errors in execution or because the result returned was empty or unexpected. Your role is to analyze the error based on the provided database schema and the details of the failed execution, and then provide a corrected version of the SQL query.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze Query Requirements:
    - Original Question: Consider what information the query is supposed to retrieve.
    - Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
    - Executed SQL Query: Review the SQL query that was previously executed and led to an error or incorrect result.
    - Execution Result: Analyze the outcome of the executed query to identify why it failed (e.g., syntax errors, incorrect column references, logical mistakes).
3. Correct the Query: 
    - Modify the SQL query to address the identified issues, ensuring it correctly fetches the requested data according to the database schema and query requirements.
    - Use the retrieved values to help write more accurate conditions when appropriate.

[IMPORTANT]
For key phrases mentioned in the question, we have provided the most similar values within the columns (TEXT-TYPE columns) denoted by "Value Examples". **This is a critical hint to identify the tables/columns that will be used in the SQL query.**

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the SQL query revision, including the detailed analysis of the previous query and the database schema, and try to fix the failed query.
</reasoning>
<result>
    The final revised SQL query that answers the question and can be executed by SQLite directly, ensure there is not any SQLite comment and not any other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Execution Result:
{RESULT}

Based on the question, table schemas, the previous query, and the execution result, analyze the result try to fix the query, and only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""

COMMON_CHECKER_PROMPT = """
# Task:
You are an SQL database expert tasked with correcting a SQL query. An external SQL checker tool has checked the SQL query and provided some suggestions to correct. Your role is to analyze the suggestions from the checker tool, and then based on the provided database schema provide a corrected version of the SQL query.

# Instructions:
1. Review Database Schema:
    - Examine the database schema to understand the database structure.
2. Analyze Query Requirements:
    - Original Question: Consider what information the query is supposed to retrieve.
    - Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
    - SQL Query: Review the SQL query that was previously checked.
    - Modification Suggestions: Review the suggestions provided by the external checker, and think how to modify the SQL to meet the suggestions.
3. Correct the Query: 
    - Modify the SQL query based the given Modification Suggestions, ensuring it correctly meet the expected suggestions.

[IMPORTANT]
Your are NOT ALLOWED to do any other modifications which are not listed in given suggestions.

# Output Format:
Please respond with XML code structured as follows.
<reasoning>
    Your detailed reasoning for the SQL query revision, including understanding the given suggestions, analyzing of the previous query and database schema, and then try to fix the query.
</reasoning>
<result>
    The final revised SQL query that answers the question and can be executed by SQLite directly, ensure there is not any SQLite comment and not any other explanation text in the SQL query.
    The SQL query must not include XML-specific characters (e.g., `&lt;`, `&gt;`, `&amp;`); only SQL-valid characters are allowed.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Modification Suggestions:
{SUGGESTIONS}

Based on the question, database schemas, previous SQL query and modification suggestions, try to fix the query, and only output the XML code (<reasoning>...</reasoning> and <result>...</result>) as your response.

# Output:
"""


def build_execution_checker_prompt(
    database_schema: str,
    question: str,
    hint: str,
    query: str,
    result: str
) -> str:
    """Build prompt for execution-based checker"""
    return EXECUTION_CHECKER_PROMPT.format(
        DATABASE_SCHEMA=database_schema,
        QUESTION=question,
        HINT=hint or "No additional hints provided.",
        QUERY=query,
        RESULT=result
    )


def build_common_checker_prompt(
    database_schema: str,
    question: str,
    hint: str,
    query: str,
    suggestions: str
) -> str:
    """Build prompt for common checker (pattern-based)"""
    return COMMON_CHECKER_PROMPT.format(
        DATABASE_SCHEMA=database_schema,
        QUESTION=question,
        HINT=hint or "No additional hints provided.",
        QUERY=query,
        SUGGESTIONS=suggestions
    )

