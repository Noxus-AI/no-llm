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


class MistralNemoConfiguration(ModelConfiguration):
    """Configuration for Mistral Nemo model"""

    identity: ModelIdentity = ModelIdentity(
        id="mistral-nemo",
        name="Mistral Nemo",
        version="2024.02",
        description="Fast and efficient version of Mistral's model optimized for quick responses.",
        creator="Mistral",
    )

    providers: Sequence[Providers] = [
        VertexProvider(model_family="mistral"),
        BedrockProvider(),
        OpenRouterProvider(),
        MistralProvider(),
        FireworksProvider(),
        GroqProvider(),
        TogetherProvider(),
    ]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=128000)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=136.4, label="Fast", description="Average (0.5-2 seconds)"),
        quality=QualityProperties(score=52.0, label="Balanced", description="Balanced Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00015, output_price_per_1k=0.00015)),
        release_date=datetime(2024, 7, 24),
        data_cutoff_date=datetime(2023, 12, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="mistral-nemo",
        litellm="vertex_ai/mistral-nemo@latest",
        langfuse="mistral-nemo",
        lmarena="mistral-nemo-latest",
        openrouter="mistralai/mistral-nemo:free",
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
