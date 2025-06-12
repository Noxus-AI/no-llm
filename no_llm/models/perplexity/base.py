from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from no_llm.config import (
    ModelConfiguration,
)


class PerplexityBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> ModelSettings:
        return super().to_pydantic_settings()
