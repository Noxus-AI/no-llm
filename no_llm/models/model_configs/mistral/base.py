from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from no_llm.models.config import (
    ModelConfiguration,
)
from no_llm.providers import (
    AnyProvider,
    BedrockProvider,
    FireworksProvider,
    GroqProvider,
    MistralProvider,
    OpenRouterProvider,
    TogetherProvider,
    VertexProvider,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class MistralBaseConfiguration(ModelConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(
        default={
            OpenRouterProvider,
            VertexProvider,
            BedrockProvider,
            FireworksProvider,
            TogetherProvider,
            MistralProvider,
            GroqProvider,
        }
    )

    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> ModelSettings:
        return super().to_pydantic_settings()
