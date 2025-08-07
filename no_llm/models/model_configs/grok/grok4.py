from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

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
from no_llm.models.config.parameters import NOT_GIVEN, NotGiven
from no_llm.models.model_configs.grok.base import GrokBaseConfiguration
from no_llm.providers import (
    GrokProvider,
    OpenRouterProvider,
    Providers,
)


class Grok4Configuration(GrokBaseConfiguration):
    identity: ModelIdentity = ModelIdentity(
        id="grok-4",
        name="Grok 4",
        version="1.0.0",
        description="Grok 4 is a powerful language model that can understand and generate natural language.",
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
        ModelCapability.REASONING,
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
        pydantic_ai="grok-4-0709",
        litellm="grok-4-0709",
        langfuse="grok-4-0709",
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
        frequency_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        presence_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        stop: ParameterValue[list[str] | NotGiven] = Field(
            default_factory=lambda: ParameterValue[list[str] | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            ),
            description="Stop sequences",
        )
        include_reasoning: ParameterValue[bool | NotGiven] = Field(
            default_factory=lambda: ParameterValue[bool | NotGiven](
                variant=ParameterVariant.UNSUPPORTED,
                value=NOT_GIVEN,
                required_capability=ModelCapability.REASONING,
            ),
            description="Whether to include reasoning steps",
        )
        reasoning_effort: ParameterValue[Literal["off", "low", "medium", "high"] | NotGiven] = Field(
            default_factory=lambda: ParameterValue[Literal["off", "low", "medium", "high"] | NotGiven](
                variant=ParameterVariant.UNSUPPORTED,
                value=NOT_GIVEN,
                required_capability=ModelCapability.REASONING,
            ),
            description="Reasoning level",
        )

    parameters: Parameters = Field(default_factory=Parameters)  # type: ignore
