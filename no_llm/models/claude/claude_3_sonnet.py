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
from no_llm.providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenRouterProvider,
    Providers,
    VertexProvider,
)


class Claude3SonnetConfiguration(ModelConfiguration):
    """Configuration for Claude 3 Sonnet model"""

    identity: ModelIdentity = ModelIdentity(
        id="claude-3-sonnet",
        name="Claude 3 Sonnet",
        version="2024.02",
        description="Claude 3 Sonnet strikes the ideal balance between intelligence and speed, particularly for enterprise workloads. It offers maximum utility at a lower price than competitors, and is engineered to be the dependable, high-endurance workhorse for scaled AI deployments.",
        creator="Anthropic",
    )

    mode: ModelMode = ModelMode.CHAT
    providers: Sequence[Providers] = [
        VertexProvider(model_family="claude"),
        BedrockProvider(),
        AnthropicProvider(),
        OpenRouterProvider(),
    ]
    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=200000, max_output_tokens=4096)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=56.2, label="Average", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=57.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.003, output_price_per_1k=0.015)),
        release_date=datetime(2024, 5, 31),
        data_cutoff_date=datetime(2023, 8, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="claude-3-sonnet",
        litellm="vertex_ai/claude-3-sonnet@20240229",
        langfuse="claude-3-sonnet",
        lmarena="claude-3-sonnet-20240229",
        openrouter="anthropic/claude-3-sonnet:free",
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
            )
        )
        seed: ParameterValue[int | NotGiven] = Field(
            default_factory=lambda: ParameterValue[int | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )

    parameters: Parameters = Field(default_factory=Parameters)  # type: ignore
