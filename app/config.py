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

    # Rewrite the question into document vocabulary before searching (1 extra small LLM call)
    query_expansion: bool = True
    # Don't wait for expansion when the question already has a strong FAISS match
    expansion_skip_min_score: float = 0.55

    # Text-to-Cypher for counting / ranking questions (1 extra LLM call, only when needed)
    text2cypher: bool = True

    # Verification: "auto" skips the 2nd LLM call when cheap checks show a well-grounded
    # answer; "always" verifies every answer; "never" turns verification off.
    verify_mode: str = "auto"
    verify_skip_min_score: float = 0.5   # best FAISS cosine score needed to skip verification

    # Cache whole answers for repeated questions (0 disables)
    answer_cache_ttl_s: int = 3600

    pdf_path: str = "data/raw/spotify_web_app_architecture.pdf"
    api_base_url: str = "http://localhost:8000"

    # Protects POST /ingest. Leave empty to allow ingest without a token (local dev only).
    admin_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()