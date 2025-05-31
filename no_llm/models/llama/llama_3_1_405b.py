from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import Field

from no_llm.config import (
    ConfigurableModelParameters,
    IntegrationAliases,
    ModelCapability,
    ModelConfiguration,
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
from no_llm.config.parameters import NotGiven
from no_llm.providers import (
    FireworksProvider,
    GroqProvider,
    OpenRouterProvider,
    Providers,
    TogetherProvider,
)


class Llama31405BConfiguration(ModelConfiguration):
    """Configuration for LLama 3.1 405B model"""

    identity: ModelIdentity = ModelIdentity(
        id="llama-3.1-405b",
        name="LLama 3.1 405B",
        version="2024.02",
        description="Newest and most advanced model from OpenAI with the most advanced performance and speed",
        creator="Meta",
    )

    providers: Sequence[Providers] = [
        TogetherProvider(),
        OpenRouterProvider(),
        FireworksProvider(),
        GroqProvider(),
    ]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=4096)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=28.3, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=72.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.GDPR, PrivacyLevel.HIPAA, PrivacyLevel.SOC2],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.01, output_price_per_1k=0.03)),
        release_date=datetime(2024, 7, 31),
        data_cutoff_date=datetime(2023, 12, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="llama-3.1-405b",
        litellm="vertex_ai/meta/llama3-405b-instruct-maas",
        langfuse="llama-3.1-405b",
        lmarena="llama-3.1-405b-instruct",
        openrouter="meta-llama/llama-3.1-405b-instruct:free",
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
