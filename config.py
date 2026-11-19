from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Advanced RAG Agent"
    environment: str = "development"
    cohere_api_key: str = ""
    cohere_chat_model: str = "command-r-plus-08-2024"
    cohere_embed_model: str = "embed-english-v3.0"
    cohere_rerank_model: str = "rerank-v3.5"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_base"
    mem0_api_key: str = ""
    mem0_enabled: bool = False
    chunk_size: int = 200
    parent_chunk_size: int = 1000
    retrieval_k: int = 20
    rerank_k: int = 5
    max_context_chars: int = 12000
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
