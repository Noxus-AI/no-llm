from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from no_llm.models.model_configs.openai.base import OpenaiBaseConfiguration
from no_llm.providers import AnyProvider, DeepseekProvider, OpenRouterProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.models.openai import OpenAIModelSettings


class DeepseekBaseConfiguration(OpenaiBaseConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(default={OpenRouterProvider, DeepseekProvider})

    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> OpenAIModelSettings:  # type: ignore
        return super().to_pydantic_settings()
