from __future__ import annotations

from typing import TYPE_CHECKING

from no_llm.config import (
    ModelConfiguration,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class MistralBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> ModelSettings:
        return super().to_pydantic_settings()
