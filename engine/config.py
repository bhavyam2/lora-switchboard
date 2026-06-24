from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    model_id: str = "Qwen/Qwen1.5-0.5B-Chat"
    lora_rank: int = 8
    lora_target_modules: list[str] = ["q_proj", "v_proj"]

    # LRU cache: how many adapters to keep hot on GPU at once
    adapter_cache_max: int = 8

    # Scheduler
    request_queue_maxsize: int = 256


settings = Settings()
