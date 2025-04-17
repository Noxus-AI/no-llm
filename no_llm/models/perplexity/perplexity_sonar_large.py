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
from no_llm.providers import OpenRouterProvider, PerplexityProvider, Providers


class PerplexitySonarLargeConfiguration(ModelConfiguration):
    """Configuration for Perplexity Sonar Large model"""

    identity: ModelIdentity = ModelIdentity(
        id="perplexity-sonar-large",
        name="Sonar Large",
        version="2024.02",
        description="Advanced online model from Perplexity with strong performance and web search integration.",
        creator="Perplexity",
    )

    providers: Sequence[Providers] = [OpenRouterProvider(), PerplexityProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.WEB_SEARCH,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=127072, max_output_tokens=127072)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=126.4, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=76.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.001, output_price_per_1k=0.001)),
        release_date=datetime(2024, 3, 1),
        data_cutoff_date=datetime(2024, 1, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="llama-3.1-sonar-large-128k-online",
        litellm="perplexity/llama-3.1-sonar-large-128k-online",
        langfuse="perplexity-sonar-large",
        lmarena="sonar-large-latest",
        openrouter="perplexity/llama-3.1-sonar-large-128k-online:free",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
            )
        )

    parameters: ConfigurableModelParameters = Field(default_factory=Parameters)
