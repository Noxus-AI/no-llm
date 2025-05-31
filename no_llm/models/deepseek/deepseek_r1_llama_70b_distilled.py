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
from no_llm.providers import GroqProvider, OpenRouterProvider, Providers


class DeepseekR1Llama70BDistilledConfiguration(ModelConfiguration):
    """Configuration for DeepSeek R1 Llama 70B Distilled model"""

    identity: ModelIdentity = ModelIdentity(
        id="deepseek-r1-llama-70b-distilled",
        name="DeepSeek R1 Distilled 70B",
        version="2024.02",
        description="DeepSeek R1 Distilled 70B model running on Groq infrastructure",
        creator="DeepSeek",
    )

    providers: Sequence[Providers] = [OpenRouterProvider(), GroqProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.REASONING,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=128000)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=180.0, label="Average", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=82.0, label="High", description="Average Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00075, output_price_per_1k=0.00099)),
        release_date=datetime(2024, 1, 1),
        data_cutoff_date=datetime(2024, 1, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="deepseek-r1-distill-llama-70b",
        litellm="groq/deepseek-r1-distill-llama-70b",
        langfuse="deepseek-r1-distill-llama-70b",
        # NOTE: not avaiable in lmarena
        lmarena="deepseek-r1",
        openrouter="deepseek/deepseek-r1-distill-llama-70b:free",
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
