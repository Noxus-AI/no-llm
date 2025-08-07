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


class GPT5NanoConfiguration(OpenaiBaseConfiguration):
    identity: ModelIdentity = ModelIdentity(
        id="gpt-5-nano",
        name="GPT 5 Nano",
        version="1.0.0",
        description="Latest OpenAI language model with strong generalist capabilities",
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
        ModelCapability.VISION,
        ModelCapability.PARALLEL_FUNCTION_CALLING,
        ModelCapability.REASONING,
    }

    constraints: ModelConstraints = ModelConstraints(
        max_input_tokens=400000,
        max_output_tokens=128000,
    )

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=121.7, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=83.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00005, output_price_per_1k=0.0004)),
        release_date=datetime(2025, 8, 6),
        data_cutoff_date=datetime(2025, 8, 6),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gpt-5-nano",
        litellm="gpt-5-nano",
        langfuse="gpt-5-nano",
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
