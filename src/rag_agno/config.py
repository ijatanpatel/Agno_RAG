from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None

    working_dir: Path = Path("./data")
    vector_db_uri: Path = Path("./data/knowledge/lancedb")
    contents_db_path: Path = Path("./data/state/knowledge.sqlite")
    state_db_path: Path = Path("./data/state/pipeline.sqlite")
    parser_output_dir: Path = Path("./data/parser_output")

    llm_model: str = "gpt-4.1-mini"
    vision_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    max_workers: int = 4
    text_chunk_size: int = 1200
    text_chunk_overlap: int = 150

    context_window_pages: int = 1
    max_context_chars: int = 5000

    content_format: str = "mineru"
    store_full_path_references: bool = False

    mcp_transport: str = "streamable-http"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000

    def ensure_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_uri.mkdir(parents=True, exist_ok=True)
        self.parser_output_dir.mkdir(parents=True, exist_ok=True)
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.contents_db_path.parent.mkdir(parents=True, exist_ok=True)