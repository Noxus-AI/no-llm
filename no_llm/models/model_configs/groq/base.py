from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr
from pydantic_ai.models.fallback import FallbackModel

from no_llm.models.config import (
    ModelConfiguration,
)
from no_llm.providers import AnyProvider, GroqProvider, OpenRouterProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class GroqBaseConfiguration(ModelConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(default={OpenRouterProvider, GroqProvider})

    def to_pydantic_model(self) -> Model:
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.models.openai import OpenAIModel

        if self.integration_aliases is None:
            msg = "Model must have integration aliases. It is required for pydantic-ai integration."
            raise TypeError(msg)
        if self.integration_aliases.pydantic_ai is None:
            msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
            raise TypeError(msg)

        models = []
        for provider in self.iter():
            if isinstance(provider, GroqProvider):
                models.append(
                    GroqModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(),
                    )
                )
            elif isinstance(provider, OpenRouterProvider):
                model_name = self.integration_aliases.openrouter or self.integration_aliases.pydantic_ai
                models.append(
                    OpenAIModel(
                        model_name=model_name,
                        provider=provider.to_pydantic(),
                    )
                )

        if len(models) == 0:
            msg = f"No compatible providers found for Groq model {self.identity.id}"
            raise RuntimeError(msg)
        return FallbackModel(
            models[0],
            *models[1:],
        )

    def to_pydantic_settings(self) -> ModelSettings:
        return super().to_pydantic_settings()
