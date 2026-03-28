from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    model_checkpoint_path: str = "ml/checkpoints/efficientnet_b0_soil.pth"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    model_config = {"env_file": ".env", "protected_namespaces": ("settings_",)}


@lru_cache
def get_settings() -> Settings:
    return Settings()
