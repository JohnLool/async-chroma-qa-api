from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    CHROMA_PATH: str = "app/database/chroma"
    EMBEDDINGS_MODEL: str = "intfloat/multilingual-e5-large"
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"

    class Config:
        env_file = ".env"


settings = Settings()