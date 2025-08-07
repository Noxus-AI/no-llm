from __future__ import annotations

from typing import TYPE_CHECKING, cast

from google.genai.types import ThinkingConfigDict
from pydantic import PrivateAttr
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModelSettings

from no_llm.models.config import (
    ModelCapability,
    ModelConfiguration,
)
from no_llm.models.config.parameters import NOT_GIVEN
from no_llm.providers import AnyProvider, GeminiProvider, OpenRouterProvider, VertexProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

THINKING_BUDGET = {
    "low": 512,
    "medium": 1024,
    "high": 4096,
}


class GeminiBaseConfiguration(ModelConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(
        default={OpenRouterProvider, VertexProvider, GeminiProvider}
    )

    def to_pydantic_model(self) -> Model:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.models.openai import OpenAIModel

        if self.integration_aliases is None:
            msg = "Model must have integration aliases. It is required for pydantic-ai integration."
            raise TypeError(msg)
        if self.integration_aliases.pydantic_ai is None:
            msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
            raise TypeError(msg)

        models = []
        for provider in self.iter():
            if isinstance(provider, VertexProvider):
                models.append(
                    GoogleModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(model_family="gemini"),  # type: ignore
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
            msg = f"No compatible providers found for Gemini model {self.identity.id}"
            raise RuntimeError(msg)
        return FallbackModel(
            models[0],
            *models[1:],
        )

    def to_pydantic_settings(self) -> GoogleModelSettings:
        base = super().to_pydantic_settings()
        # nbase = cast(dict, {f"google_{k}": v for k, v in base.items()})
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        if ModelCapability.REASONING not in self.capabilities:
            return GoogleModelSettings(**base)

        elif reasoning_effort in ["off", NOT_GIVEN]:
            include_thoughts = False
            thinking_budget = 0
        else:
            include_thoughts = True
            thinking_budget = THINKING_BUDGET[reasoning_effort]  # type: ignore
        return GoogleModelSettings(
            **base,
            google_thinking_config=ThinkingConfigDict(
                include_thoughts=include_thoughts, thinking_budget=thinking_budget
            ),
        )
