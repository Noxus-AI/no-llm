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
from no_llm.config.parameters import NOT_GIVEN, NotGiven
from no_llm.providers import AzureProvider, OpenAIProvider, OpenRouterProvider, Providers


class GPT4Configuration(ModelConfiguration):
    """Configuration for GPT-4 model"""

    identity: ModelIdentity = ModelIdentity(
        id="gpt-4",
        name="GPT-4",
        version="1.0.0",
        description="Advanced language model with strong reasoning capabilities",
        creator="OpenAI",
    )

    providers: Sequence[Providers] = [AzureProvider(), OpenRouterProvider(), OpenAIProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=8000, max_output_tokens=4000)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=75.5, label="Medium", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=95.0, label="Exceptional", description="Best-in-class Performance"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.03, output_price_per_1k=0.06)),
        release_date=datetime(2024, 1, 25),
        data_cutoff_date=datetime(2023, 12, 31),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gpt-4o", litellm="gpt-4o", langfuse="gpt-4", openrouter="openai/gpt-4-turbo"
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
        top_p: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](variant=ParameterVariant.FIXED, value=1.0)
        )
        top_k: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        frequency_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](variant=ParameterVariant.FIXED, value=0.0)
        )
        presence_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](variant=ParameterVariant.FIXED, value=0.0)
        )
        max_tokens: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](variant=ParameterVariant.FIXED, value=4096)
        )
        stop: ParameterValue[list[str] | NotGiven] = Field(
            default_factory=lambda: ParameterValue[list[str] | NotGiven](
                variant=ParameterVariant.VARIABLE, value=NOT_GIVEN
            )
        )
        seed: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )

    parameters: ConfigurableModelParameters = Field(default_factory=Parameters)
