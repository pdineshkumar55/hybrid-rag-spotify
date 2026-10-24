from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = ""  # empty = the user's home database (on Aura: the instance ID)

    faiss_index_dir: str = "vectorstore"
    pdf_path: str = "data/raw/spotify_web_app_architecture.pdf"
    api_base_url: str = "http://localhost:8000"

    # Protects POST /ingest. Leave empty to allow ingest without a token (local dev only).
    admin_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()