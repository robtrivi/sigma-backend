from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import FieldValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    project_name: str = "SIGMA Backend API"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://ejemplo:ejemplo@localhost:5432/ejemplo"
    postgis_enabled: bool = True
    data_dir: Path = Path("data")
    scenes_dir: Path | None = None
    reports_dir: Path | None = None
    green_class_ids: List[str] | str = ["green", "tree_canopy", "park"]
    default_epsg: int = 4326
    segmentation_model_path: Path = Path("model.h5")

    @field_validator("green_class_ids", mode="before")
    @classmethod
    def parse_green_class_ids(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v

    @field_validator("scenes_dir", mode="before")
    @classmethod
    def default_scenes_dir(cls, v: Path | None, info: FieldValidationInfo) -> Path:
        if v is not None:
            return Path(v)
        base = info.data.get("data_dir", Path("data"))
        return Path(base) / "scenes"

    @field_validator("reports_dir", mode="before")
    @classmethod
    def default_reports_dir(cls, v: Path | None, info: FieldValidationInfo) -> Path:
        if v is not None:
            return Path(v)
        base = info.data.get("data_dir", Path("data"))
        return Path(base) / "reports"

    @field_validator("data_dir", mode="after")
    @classmethod
    def ensure_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        (v / "scenes").mkdir(parents=True, exist_ok=True)
        (v / "reports").mkdir(parents=True, exist_ok=True)
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
