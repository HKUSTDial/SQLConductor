"""
Configuration for the inference preprocessing pipeline.
Loads settings from a TOML config file.
"""

import tomllib
import threading
from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from loguru import logger


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "workspace" / "preprocess"


class LLMConfig(BaseModel):
    """LLM configuration for keyword extraction."""
    model: str = Field(..., description="The model name")
    base_url: str = Field(..., description="The base url of the model service")
    api_key: str = Field(..., description="The api key of the model service")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    api_type: Literal["openai", "azure"] = Field(default="openai", description="API type")
    api_version: Optional[str] = Field(default=None, description="Azure API version")
    fix_end_token: bool = Field(default=False, description="Whether to fix missing end tokens in LLM response")
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(default=None, description="Reasoning effort for reasoning models")


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    type: Literal["spider", "bird"] = Field(..., description="Dataset type")
    split: Literal["dev", "test"] = Field(..., description="Dataset split")
    root_path: str = Field(..., description="Root path of the dataset")
    save_path: str = Field(default="", description="Save path for preprocessed dataset pkl")
    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples to load")
    table_filter: Optional[str] = Field(
        default=None,
        description=(
            "Path to a JSON file mapping db_id -> list of table names to keep. "
            "Tables not listed are excluded from schema loading, vector DB "
            "indexing, and value retrieval. The SQLite databases are NOT "
            "modified, so generated SQL can still reference any table. "
            "Useful for large databases (e.g. Beaver NW) where the full "
            "schema exceeds the model context window."
        ),
    )
    per_question_table_filter: Optional[str] = Field(
        default=None,
        description=(
            "Path to a JSON file mapping question_id (str) -> list of table names "
            "allowed for that specific question. Each question's Task will have its "
            "table_schema_dict restricted to these tables, preventing context-window "
            "overflow for databases with extremely large number of tables (e.g. Beaver NW). "
        ),
    )

    @model_validator(mode="after")
    def validate_split(self):
        if self.type == "spider":
            if self.split not in ["dev", "test"]:
                raise ValueError(f"Invalid split for spider: {self.split}")
        elif self.type == "bird":
            if self.split not in ["dev"]:
                raise ValueError(f"Invalid split for bird: {self.split}")
        return self


class VectorDatabaseConfig(BaseModel):
    """Vector database (ChromaDB) configuration."""
    api_type: Literal["local", "openai"] = Field(default="local", description="Embedding API type")
    embedding_model_name_or_path: str = Field(..., description="Embedding model name or path")
    use_qwen3_embedding: bool = Field(default=False, description="Use Qwen3 embedding function")
    local_files_only: bool = Field(default=False, description="Use local model files only")
    normalize_embeddings: bool = Field(default=False, description="Normalize embeddings")
    base_url: Optional[str] = Field(default=None, description="Embedding API base url")
    api_key: Optional[str] = Field(default=None, description="Embedding API key")
    store_root_path: str = Field(default="", description="Root path for vector database storage")
    max_value_length: int = Field(default=100, description="Maximum length of database values to index")
    batch_size: int = Field(default=1024, description="Batch size for adding documents")
    n_parallel: int = Field(default=1, description="Parallel threads for vector DB creation")
    lower_meta_data: bool = Field(default=True, description="Lowercase metadata in vector DB")


class ValueRetrievalConfig(BaseModel):
    """Value retrieval configuration."""
    llm: LLMConfig = Field(..., description="LLM config for keyword extraction")
    n_results: int = Field(default=5, description="Number of results to retrieve per column")
    n_parallel: int = Field(default=16, description="Parallel threads for value retrieval")
    save_path: str = Field(default="", description="Save path for value retrieval results")
    max_retry: int = Field(default=3, description="Max retries for LLM extraction")


class OutputConfig(BaseModel):
    """Output configuration for final Task pkl."""
    save_path: str = Field(default="", description="Save path for the final Task pkl file")
    max_example_length: int = Field(default=100, description="Maximum length of value examples in schema context")


class PreprocessConfig(BaseModel):
    """Top-level preprocessing pipeline configuration."""
    dataset: DatasetConfig
    vector_database: VectorDatabaseConfig
    value_retrieval: ValueRetrievalConfig
    output: OutputConfig = Field(default_factory=OutputConfig)


class PreprocessConfigManager:
    """Singleton config manager that loads from TOML file."""
    _instance = None
    _lock = threading.Lock()
    _config: Optional[PreprocessConfig] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str):
        """Load configuration from a TOML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "rb") as f:
            raw = tomllib.load(f)

        dataset_cfg = raw.get("dataset", {})
        vector_db_cfg = raw.get("vector_database", {})
        value_retrieval_cfg = raw.get("value_retrieval", {})
        output_cfg = raw.get("output", {})

        vr_llm_raw = value_retrieval_cfg.pop("llm", {})
        vr_llm = LLMConfig(**vr_llm_raw)

        self._config = PreprocessConfig(
            dataset=DatasetConfig(**dataset_cfg),
            vector_database=VectorDatabaseConfig(**vector_db_cfg),
            value_retrieval=ValueRetrievalConfig(llm=vr_llm, **value_retrieval_cfg),
            output=OutputConfig(**output_cfg),
        )
        logger.info(f"Preprocessing config loaded from {config_path}")
        return self._config

    @property
    def config(self) -> PreprocessConfig:
        if self._config is None:
            raise RuntimeError("Config not loaded. Call load(config_path) first.")
        return self._config

    @property
    def dataset_config(self) -> DatasetConfig:
        return self.config.dataset

    @property
    def vector_database_config(self) -> VectorDatabaseConfig:
        return self.config.vector_database

    @property
    def value_retrieval_config(self) -> ValueRetrievalConfig:
        return self.config.value_retrieval

    @property
    def output_config(self) -> OutputConfig:
        return self.config.output
