from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic import PrivateAttr
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from no_llm.models.config import (
    ModelCapability,
    ModelConfiguration,
)
from no_llm.models.config.parameters import NOT_GIVEN
from no_llm.providers import AnyProvider, AzureProvider, OpenAIProvider, OpenRouterProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class OpenaiBaseConfiguration(ModelConfiguration):
    _compatible_providers: set[type[AnyProvider]] = PrivateAttr(
        default={AzureProvider, OpenRouterProvider, OpenAIProvider}
    )

    def to_pydantic_model(self) -> Model:
        from pydantic_ai.models.openai import OpenAIModel, OpenAIResponsesModel

        if self.integration_aliases is None:
            msg = "Model must have integration aliases. It is required for pydantic-ai integration."
            raise TypeError(msg)
        if self.integration_aliases.pydantic_ai is None:
            msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
            raise TypeError(msg)

        models = []
        for provider in self.iter():
            if isinstance(provider, OpenAIProvider):
                models.append(
                    OpenAIResponsesModel(
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
            elif isinstance(provider, AzureProvider):
                models.append(
                    OpenAIModel(
                        model_name=self.integration_aliases.pydantic_ai,
                        provider=provider.to_pydantic(),
                    )
                )
        if len(models) == 0:
            msg = f"No compatible providers found for OpenAI model {self.identity.id}"
            raise RuntimeError(msg)
        return FallbackModel(
            models[0],
            *models[1:],
        )

    def to_pydantic_settings(self) -> OpenAIResponsesModelSettings:
        base = super().to_pydantic_settings()
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        # nbase = cast(dict, {f"openai_{k}": v for k, v in base.items()})
        if ModelCapability.REASONING in self.capabilities and reasoning_effort not in [
            None,
            "off",
            NOT_GIVEN,
        ]:
            return OpenAIResponsesModelSettings(
                **base,
                openai_reasoning_effort=reasoning_effort,  # type: ignore
                openai_reasoning_summary="detailed",
            )  # type: ignore
        return OpenAIResponsesModelSettings(**base)
