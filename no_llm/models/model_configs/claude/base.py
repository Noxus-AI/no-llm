from __future__ import annotations

from typing import TYPE_CHECKING, cast

from anthropic.types.beta import (
    BetaThinkingConfigDisabledParam,
    BetaThinkingConfigEnabledParam,
)
from pydantic import PrivateAttr
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.fallback import FallbackModel

from no_llm.models.config import (
    ModelCapability,
    ModelConfiguration,
)
from no_llm.models.config.parameters import NOT_GIVEN
from no_llm.providers import AnthropicProvider, AnyProvider, BedrockProvider, OpenRouterProvider, VertexProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

THINKING_BUDGET = {
    "low": 1025,
    "medium": 2048,
    "high": 4096,
}


class ClaudeBaseConfiguration(ModelConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(
        default={AnthropicProvider, BedrockProvider, OpenRouterProvider, VertexProvider}
    )

    def to_pydantic_model(self) -> Model:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.openai import OpenAIModel

        if self.integration_aliases is None:
            msg = "Model must have integration aliases. It is required for pydantic-ai integration."
            raise TypeError(msg)
        if self.integration_aliases.pydantic_ai is None:
            msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
            raise TypeError(msg)

        models = []
        for provider in self.iter():
            if isinstance(provider, AnthropicProvider):
                models.append(
                    AnthropicModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(),
                    )
                )
            elif isinstance(provider, VertexProvider):
                models.append(
                    AnthropicModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(model_family="claude"),  # type: ignore
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
            elif isinstance(provider, BedrockProvider):
                models.append(
                    AnthropicModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(),  # type: ignore
                    )
                )

        if len(models) == 0:
            msg = f"No compatible providers found for Claude model {self.identity.id}"
            raise RuntimeError(msg)
        return FallbackModel(
            models[0],
            *models[1:],
        )

    def to_pydantic_settings(self) -> AnthropicModelSettings:
        base = super().to_pydantic_settings()
        # nbase = cast(dict, {f"anthropic_{k}": v for k, v in base.items()})
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        if ModelCapability.REASONING not in self.capabilities:
            return AnthropicModelSettings(**base)
        elif reasoning_effort in ["off", NOT_GIVEN]:
            thinking_config = BetaThinkingConfigDisabledParam(type="disabled")
        else:
            # NOTE: for reasoning temperature needs to be 1
            base["temperature"] = 1.0
            # NOTE: for reasoning max_tokens needs to be at least the thinking budget
            base["max_tokens"] = max(base["max_tokens"], THINKING_BUDGET[reasoning_effort]) + 1  # type: ignore
            thinking_config = BetaThinkingConfigEnabledParam(
                type="enabled", budget_tokens=THINKING_BUDGET[reasoning_effort]
            )
        return AnthropicModelSettings(
            **base,
            anthropic_thinking=thinking_config,
        )
