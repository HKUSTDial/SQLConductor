from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import chardet
import pandas as pd
from loguru import logger

from src.database.sql_execution import execute_sql_without_timeout

def lower_str_list(str_list: List[Any]) -> List[Any]:
    """
    Convert a list of strings or nested lists to a list of lowercase strings or nested lists.
    
    Args:
        str_list: List of strings or nested lists.
    Returns:
        List of lowercase strings or nested lists.
    """
    return [s.lower() if isinstance(s, str) else lower_str_list(s) for s in str_list]

def load_table_names(db_id: str, database_root_dir: str) -> List[str]:
    """
    Load table names from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        List of table names.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    execution_result = execute_sql_without_timeout(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
    if execution_result.result is None:
        logger.warning(f"Could not load table names for db_id: {db_id}. The database file might be missing or corrupted at {db_path}")
        return [] 
    table_names = execution_result.result
    return [row[0].strip() for row in table_names]

def load_column_names_and_types(db_id: str, database_root_dir: str, table_name: str) -> List[Tuple[str, str]]:
    """
    Load column names and types from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of tuples, each containing a column name and its type.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_info = execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{table_name}`);").result
    return [(row[1].strip(), row[2].strip()) for row in table_info]

def load_primary_keys(db_id: str, database_root_dir: str, table_name: str) -> List[str]:
    """
    Load primary keys from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of primary key column names.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    table_info = execute_sql_without_timeout(db_path, f"PRAGMA table_info(`{table_name}`);").result
    return [row[1].strip() for row in table_info if row[5] != 0]

def load_foreign_keys(db_id: str, database_root_dir: str, table_name: str) -> List[Tuple[str, str, str, str]]:
    """
    Load foreign keys from a table.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
    Returns:
        List of tuples, each containing a source table name, source column name, target table name, and target column name.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    foreign_keys_list = execute_sql_without_timeout(db_path, f"PRAGMA foreign_key_list(`{table_name}`);").result
    
    # Foreign key metadata overrides.
    spcial_cases_for_bird_trian_databases = {
        ("works_cycles", "SalesOrderHeader", "ShipMethodID", "Address", "AddressID"): ("SalesOrderHeader", "ShipMethodID", "ShipMethod", "ShipMethodID"),
        ("mondial_geo", "city", "Province", "province", None): ("city", "Province", "province", "Name"),
        ("mondial_geo", "geo_desert", "Province", "province", None): ("geo_desert", "Province", "province", "Name"),
        ("mondial_geo", "geo_estuary", "Province", "province", None): ("geo_estuary", "Province", "province", "Name"),
        ("mondial_geo", "geo_island", "Province", "province", None): ("geo_island", "Province", "province", "Name"),
        ("mondial_geo", "geo_lake", "Province", "province", None): ("geo_lake", "Province", "province", "Name"),
        ("mondial_geo", "geo_mountain", "Province", "province", None): ("geo_mountain", "Province", "province", "Name"),
        ("mondial_geo", "geo_river", "Province", "province", None): ("geo_river", "Province", "province", "Name"),
        ("mondial_geo", "geo_sea", "Province", "province", None): ("geo_sea", "Province", "province", "Name"),
        ("mondial_geo", "geo_source", "Province", "province", None): ("geo_source", "Province", "province", "Name"),
        ("mondial_geo", "located", "Province", "province", None): ("located", "Province", "province", "Name"),
        ("mondial_geo", "located", "City", "city", None): ("located", "City", "city", "Name"),
        ("mondial_geo", "locatedOn", "Province", "province", None): ("locatedOn", "Province", "province", "Name"),
        ("mondial_geo", "locatedOn", "City", "city", None): ("locatedOn", "City", "city", "Name"),
        ("mondial_geo", "organization", "Province", "province", None): ("organization", "Province", "province", "Name"),
        ("mondial_geo", "organization", "City", "city", None): ("organization", "City", "city", "Name")
    }

    foreign_keys = []
    for foreign_key in foreign_keys_list:
        source_table_name = table_name.strip()
        source_column_name = foreign_key[3].strip()
        target_table_name = foreign_key[2].strip()
        target_column_name = None
        if foreign_key[4] is not None:
            target_column_name = foreign_key[4].strip()
        
        current_case = (db_id, source_table_name, source_column_name, target_table_name, target_column_name)
        if current_case in spcial_cases_for_bird_trian_databases:
            source_table_name, source_column_name, target_table_name, target_column_name = spcial_cases_for_bird_trian_databases[current_case]
        
        if target_column_name is None:
            # Resolve omitted target columns from primary keys.
            target_table_primary_keys = load_primary_keys(db_id, database_root_dir, target_table_name)
            if len(target_table_primary_keys) >= 1:
                target_column_name = target_table_primary_keys[0].strip()
                logger.debug(f"Resolved foreign key target: {target_table_name}.{target_column_name} for {source_table_name}.{source_column_name}")
            else:
                raise ValueError(f"Target column is None and CAN NOT be fixed by primary keys of target table {target_table_name}, source table: {source_table_name}, source column: {source_column_name}")
        foreign_keys.append((source_table_name, source_column_name, target_table_name, target_column_name))
    return foreign_keys

def _normalize_description_string(description: str) -> str:
    """
    Normalize the description string.
    """
    description = description.replace("\r", "").replace("\n", "").replace("commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description

def load_database_description(db_id: str, database_root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load database description from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        A dictionary with lowercased table names as keys and table descriptions as values.
        The table description is a dictionary with lowercased original column names as keys and column descriptions as values.
    """
    db_description_dir = Path(database_root_dir) / db_id / "database_description"
    if not db_description_dir.exists():
        logger.debug(f"Database description for database {db_id} does not exist, skipping...")
        return {}
    database_description = {}
    for csv_file in db_description_dir.glob("*.csv"):
        table_name_lower = csv_file.stem.lower().strip()
        encoding_type = chardet.detect(csv_file.read_bytes())["encoding"]
        table_description = {}
        table_description_df = pd.read_csv(csv_file, encoding=encoding_type, index_col=False)
        for _, row in table_description_df.iterrows():
            if pd.isna(row["original_column_name"]):
                continue
            original_column_name_lower = row["original_column_name"].strip().lower()
            expanded_column_name = row["column_name"].strip() if pd.notna(row["column_name"]) else ""
            column_description = _normalize_description_string(row["column_description"]) if pd.notna(row["column_description"]) else ""
            data_format = row["data_format"].strip() if pd.notna(row["data_format"]) else ""
            value_description = _normalize_description_string(row["value_description"]) if pd.notna(row["value_description"]) else ""
            if value_description.lower().startswith("not useful"):
                value_description = value_description[len("not useful"):].strip()
            table_description[original_column_name_lower] = {
                "original_column_name_lower": original_column_name_lower,
                "expanded_column_name": expanded_column_name,
                "column_description": column_description,
                "data_format": data_format,
                "value_description": value_description
            }
        database_description[table_name_lower] = table_description
    return database_description

def load_value_examples(db_id: str, database_root_dir: str, table_name: str, column_name: str, max_num_examples: int = 3, max_example_length: int = 50) -> List[str]:
    """
    Load value examples from database.
    
    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
        table_name: Table name.
        column_name: Column name.
        max_num_examples: Maximum number of value examples to load.
        max_example_length: Maximum length of value examples.
    Returns:
        List of value examples.
    """
    db_path = Path(database_root_dir) / db_id / f"{db_id}.sqlite"
    examples = execute_sql_without_timeout(db_path, f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL AND `{column_name}` != '' AND length(cast(`{column_name}` as text)) <= {max_example_length} LIMIT {max_num_examples};").result
    return [example[0] for example in examples]

def load_database_schema_dict(db_id: str, database_root_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load the database schema as a dictionary.

    Args:
        db_id: Database ID.
        database_root_dir: Root directory of databases.
    Returns:
        A dictionary representing the database schema.
    """
    database_schema_dict = {"db_id": db_id, "db_directory": Path(database_root_dir) / db_id, "tables": {}}
    table_names = load_table_names(db_id, database_root_dir)
    for table_name in table_names:
        table_schema_dict = {}
        table_schema_dict["table_name"] = table_name
        table_schema_dict["columns"] = {}
        
        primary_keys = load_primary_keys(db_id, database_root_dir, table_name)
        
        foreign_keys = load_foreign_keys(db_id, database_root_dir, table_name)
        
        database_description = load_database_description(db_id, database_root_dir)
        
        column_names_and_types = load_column_names_and_types(db_id, database_root_dir, table_name)
        for column_name, column_type in column_names_and_types:
            column_schema_dict = {}
            column_schema_dict["original_column_name"] = column_name
            column_schema_dict["column_type"] = column_type
            column_schema_dict["foreign_keys"] = []
            column_schema_dict["referenced_by"] = []
            
            if column_name.lower() in lower_str_list(primary_keys):
                column_schema_dict["primary_key"] = True
            else:
                column_schema_dict["primary_key"] = False
            
            for source_table_name, source_column_name, target_table_name, target_column_name in foreign_keys:
                if source_table_name.lower() == table_name.lower() and source_column_name.lower() == column_name.lower():
                    column_schema_dict["foreign_keys"].append((target_table_name, target_column_name))
                if target_table_name.lower() == table_name.lower() and target_column_name.lower() == column_name.lower():
                    column_schema_dict["referenced_by"].append((source_table_name, source_column_name))
                    
            if table_name.lower() in database_description and column_name.lower() in database_description[table_name.lower()]:
                column_schema_dict["expanded_column_name"] = database_description[table_name.lower()][column_name.lower()]["expanded_column_name"]
                column_schema_dict["column_description"] = database_description[table_name.lower()][column_name.lower()]["column_description"]
                column_schema_dict["value_description"] = database_description[table_name.lower()][column_name.lower()]["value_description"]
            
            if column_type.upper() != "BLOB":
                column_schema_dict["value_examples"] = load_value_examples(db_id, database_root_dir, table_name, column_name)
            else:
                column_schema_dict["value_examples"] = []
            
            table_schema_dict["columns"][column_name] = column_schema_dict
        database_schema_dict["tables"][table_name] = table_schema_dict
    return database_schema_dict

def build_table_profile_statement(table_schema_dict: Dict[str, Dict[str, Any]],
                              add_expanded_column_name: bool = True,
                              add_column_description: bool = True,
                              add_value_description: bool = True,
                              add_value_examples: bool = True,
                              max_example_length: int = 100) -> str:
    """
    Build the DDL statement for a table.
    
    Args:
        table_schema_dict: The table schema dictionary.
        add_expanded_column_name: Whether to add the expanded column name.
        add_column_description: Whether to add the column description.
        add_value_description: Whether to add the value description.
        add_value_examples: Whether to add the value examples.
        max_example_length: Maximum length of value examples.
    Returns:
        The DDL statement for the table.
    """
    table_name = table_schema_dict['table_name']
    statement = f"- Table: `{table_name}`\n[\n"
    
    column_profiles = []
    columns = list(table_schema_dict["columns"].items())
    pk_columns = [(col_name, col_schema) for col_name, col_schema in columns if col_schema.get("primary_key", False)]
    non_pk_columns = [(col_name, col_schema) for col_name, col_schema in columns if not col_schema.get("primary_key", False)]
    ordered_columns = pk_columns + non_pk_columns
    
    primary_keys = []
    foreign_keys = []
    
    for column_name, column_schema in ordered_columns:
        column_profile = f"`{column_name}`: {column_schema['column_type']}"
        
        if column_schema.get("primary_key", False):
            column_profile += f" | Primary Key"
            primary_keys.append(f"`{column_name}`")
            
        comment_parts = []
        if add_expanded_column_name and column_schema.get("expanded_column_name", "").strip() != "" and column_schema.get("expanded_column_name", "").strip().lower() != column_name.lower():
            comment_parts.append(f"Expanded Name: {column_schema['expanded_column_name']}")
        if add_column_description and column_schema.get("column_description", "").strip() != "" and column_schema.get("column_description", "").strip().lower() != column_name.lower() and column_schema.get("column_description", "").strip().lower() != column_schema.get("expanded_column_name", "").strip().lower():
            comment_parts.append(f"Description: {column_schema['column_description']}")
        if add_value_description and column_schema.get("value_description", "").strip() != "":
            comment_parts.append(f"Value Description: {column_schema['value_description']}")
            
        if comment_parts:
            column_profile += f" | {' | '.join(comment_parts)}"
            
        if add_value_examples and column_schema.get("value_examples") and len(column_schema["value_examples"]) > 0:
            valid_examples = [val for val in column_schema['value_examples'] if len(str(val)) <= max_example_length]
            if valid_examples:
                column_profile += f" | Value Examples: {valid_examples}"
                
        column_profiles.append(f"({column_profile})")
        
        for foreign_key in column_schema.get("foreign_keys", []):
            foreign_keys.append((column_name, *foreign_key))
            
    joined_profiles = ',\n'.join(column_profiles)
    statement += f"{joined_profiles}\n]\n"
    
    if primary_keys:
        statement += "Primary Key:\n"
        statement += f"({', '.join(primary_keys)})\n"
        
    if foreign_keys:
        fks_str = []
        for source_col, target_table, target_col in foreign_keys:
            fks_str.append(f"`{table_name}`.`{source_col}` = `{target_table}`.`{target_col}`")
        joined_fks = '\n'.join(fks_str)
        statement += "Foreign Keys:\n"
        statement += f"{joined_fks}\n"
        
    return statement.strip()

def map_lower_table_name_to_original_table_name(lower_table_name: str, database_schema_dict: Dict[str, Any]) -> Optional[str]:
    """
    Map a lower case table name to the original table name.
    
    Args:
        lower_table_name: The lower case table name.
        database_schema_dict: The database schema dictionary.
    Returns:
        The original table name.
    """
    for table_name in database_schema_dict["tables"].keys():
        if table_name.lower() == lower_table_name.lower():
            return table_name
    return None

def map_lower_column_name_to_original_column_name(original_table_name: str, lower_column_name: str, database_schema_dict: Dict[str, Any]) -> Optional[str]:
    """
    Map a lower case column name to the original column name.
    
    Args:
        original_table_name: The original table name.
        lower_column_name: The lower case column name.
        database_schema_dict: The database schema dictionary.
    Returns:
        The original column name.
    """
    if original_table_name not in database_schema_dict["tables"]:
        return None
    for column_name in database_schema_dict["tables"][original_table_name]["columns"].keys():
        if column_name.lower() == lower_column_name.lower():
            return column_name
    return None
