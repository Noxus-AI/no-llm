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
    BedrockProvider,
    FireworksProvider,
    GroqProvider,
    MistralProvider,
    OpenRouterProvider,
    Providers,
    TogetherProvider,
    VertexProvider,
)


class MistralLargeConfiguration(ModelConfiguration):
    """Configuration for Mistral Large model"""

    identity: ModelIdentity = ModelIdentity(
        id="mistral-large",
        name="Mistral Large",
        version="2024.02",
        description="Latest version of Mistral's large model with advanced capabilities for complex tasks.",
        creator="Mistral",
    )

    providers: Sequence[Providers] = [
        VertexProvider(model_family="mistral"),
        BedrockProvider(),
        OpenRouterProvider(),
        FireworksProvider(),
        TogetherProvider(),
        MistralProvider(),
        GroqProvider(),
    ]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=8191)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=38.4, label="Average", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=73.0, label="High", description="Strong General Performance"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.002, output_price_per_1k=0.006)),
        release_date=datetime(2024, 7, 24),
        data_cutoff_date=datetime(2023, 12, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="mistral-large-2411",
        litellm="vertex_ai/mistral-large-2411",
        langfuse="mistral-large-2411",
        lmarena="mistral-large-latest",
        openrouter="mistralai/mistral-large-2411:free",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=1.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
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
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )
        presence_penalty: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.UNSUPPORTED, value=NOT_GIVEN
            )
        )

    parameters: Parameters = Field(default_factory=Parameters)  # type: ignore
