from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from no_llm.models.config import (
    ConfigurableModelParameters,
    IntegrationAliases,
    ModelCapability,
    ModelConstraints,
    ModelIdentity,
    ModelMetadata,
    ModelMode,
    ModelPricing,
    ModelProperties,
    ParameterValue,
    ParameterVariant,
    PrivacyLevel,
    QualityProperties,
    RangeValidation,
    SpeedProperties,
    TokenPrices,
)
from no_llm.models.config.parameters import NotGiven
from no_llm.models.model_configs.grok.base import GrokBaseConfiguration
from no_llm.providers import (
    GrokProvider,
    OpenRouterProvider,
    Providers,
)


class Grok3Configuration(GrokBaseConfiguration):
    identity: ModelIdentity = ModelIdentity(
        id="grok-3",
        name="Grok 3",
        version="1.0.0",
        description="Grok 3 is a powerful language model that can understand and generate natural language.",
        creator="Grok",
    )

    providers: Sequence[Providers] = [GrokProvider(), OpenRouterProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
        ModelCapability.PARALLEL_FUNCTION_CALLING,
    }

    constraints: ModelConstraints = ModelConstraints(
        max_input_tokens=1047576,
        max_output_tokens=32768,
    )

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=121.7, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=83.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.002, output_price_per_1k=0.008)),
        release_date=datetime(2024, 1, 25),
        data_cutoff_date=datetime(2023, 12, 31),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="grok-3",
        litellm="grok-3",
        langfuse="grok-3",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
            )
        )

    parameters: Parameters = Field(default_factory=Parameters)  # type: ignore
