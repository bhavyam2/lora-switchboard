from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    model_id: str = "EleutherAI/pythia-70m"
    lora_rank: int = 8
    lora_target_modules: list[str] = ["query_key_value"]  # pythia uses fused QKV

    # LRU cache: how many adapters to keep hot on GPU at once
    adapter_cache_max: int = 8

    # Scheduler
    request_queue_maxsize: int = 256


settings = Settings()
