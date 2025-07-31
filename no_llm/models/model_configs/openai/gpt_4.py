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
from no_llm.models.model_configs.openai.base import OpenaiBaseConfiguration
from no_llm.providers import (
    AzureProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Providers,
)


class GPT4Configuration(OpenaiBaseConfiguration):
    """Configuration for GPT-4 model"""

    identity: ModelIdentity = ModelIdentity(
        id="gpt-4",
        name="GPT-4",
        version="1.0.0",
        description="Advanced language model with strong reasoning capabilities",
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
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.PARALLEL_FUNCTION_CALLING,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=4096)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=75.5, label="Medium", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=95.0, label="Exceptional", description="Best-in-class Performance"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.01, output_price_per_1k=0.03)),
        release_date=datetime(2024, 1, 25),
        data_cutoff_date=datetime(2023, 12, 31),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gpt-4-turbo",
        litellm="gpt-4-turbo",
        langfuse="gpt-4-turbo",
        openrouter="openai/gpt-4-turbo",
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
