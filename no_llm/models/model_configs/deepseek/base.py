from __future__ import annotations

from typing import TYPE_CHECKING

from no_llm.models.model_configs.openai.base import OpenaiBaseConfiguration

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.models.openai import OpenAIModelSettings


class DeepseekBaseConfiguration(OpenaiBaseConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> OpenAIModelSettings:  # type: ignore
        return super().to_pydantic_settings()
