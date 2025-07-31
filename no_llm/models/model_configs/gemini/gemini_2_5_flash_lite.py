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
    QualityProperties,
    RangeValidation,
    SpeedProperties,
    TokenPrices,
)
from no_llm.models.config.parameters import NOT_GIVEN, NotGiven
from no_llm.models.model_configs.gemini.base import GeminiBaseConfiguration
from no_llm.providers import OpenRouterProvider, Providers, VertexProvider


class Gemini25FlashLiteConfiguration(GeminiBaseConfiguration):
    """Configuration for Gemini 2.5 Flash Lite model"""

    identity: ModelIdentity = ModelIdentity(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash Lite",
        version="2025.04",
        description="Lite version of Gemini 2.5 Flash, optimized for rapid responses while maintaining strong performance across multimodal tasks.",
        creator="Google",
    )

    providers: Sequence[Providers] = [
        VertexProvider(model_family="gemini"),
        OpenRouterProvider(),
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
        ModelCapability.AUDIO_TRANSCRIPTION,
        ModelCapability.VIDEO_TRANSCRIPTION,
        ModelCapability.REASONING,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=1048576, max_output_tokens=65535)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=248.6, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=76.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.0001, output_price_per_1k=0.0004)),
        release_date=datetime(2025, 6, 17),
        data_cutoff_date=datetime(2025, 1, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gemini-2.5-flash-lite-preview-06-17",
        litellm="gemini/gemini-2.5-flash-lite-preview-06-17",
        langfuse="gemini-2.5-flash-lite-preview-06-17",
        lmarena="gemini-2.5-flash-lite-preview-06-17",
        openrouter="google/gemini-2.5-flash-lite-preview-06-17",
    )

    class Parameters(ConfigurableModelParameters):
        model_config = ConfigurableModelParameters.model_config
        temperature: ParameterValue[float | NotGiven] = Field(
            default_factory=lambda: ParameterValue[float | NotGiven](
                variant=ParameterVariant.VARIABLE,
                value=1.5,
                validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
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
