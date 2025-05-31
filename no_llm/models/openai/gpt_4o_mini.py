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
    AzureProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Providers,
)


class GPT4OMiniConfiguration(ModelConfiguration):
    """Configuration for GPT-4o Mini model"""

    identity: ModelIdentity = ModelIdentity(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        version="2024.02",
        description="Smaller and faster version of GPT-4o optimized for quick responses while maintaining good quality.",
        creator="OpenAI",
    )

    providers: Sequence[Providers] = [
        AzureProvider(),
        OpenRouterProvider(),
        OpenAIProvider(),
    ]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.TOOLS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
        ModelCapability.PARALLEL_FUNCTION_CALLING,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=16384)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=111.4, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=72.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00015, output_price_per_1k=0.0006)),
        release_date=datetime(2024, 5, 13),
        data_cutoff_date=datetime(2023, 10, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gpt-4o-mini",
        litellm="gpt-4o-mini",
        langfuse="gpt-4o-mini",
        openrouter="openai/gpt-4o-mini-latest",
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
