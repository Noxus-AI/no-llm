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
    QualityProperties,
    RangeValidation,
    SpeedProperties,
    TokenPrices,
)
from no_llm.config.parameters import NOT_GIVEN, NotGiven
from no_llm.providers import OpenRouterProvider, Providers, VertexProvider


class Gemini15FlashConfiguration(ModelConfiguration):
    """Configuration for Gemini 1.5 Flash model"""

    identity: ModelIdentity = ModelIdentity(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        version="2024.02",
        description="Gemini 1.5 Flash is a smaller and faster version of Google's best foundation model, performing with respectable quality at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video.",
        creator="Google",
    )

    providers: Sequence[Providers] = [VertexProvider(), OpenRouterProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.VISION,
    }

    constraints: ModelConstraints = ModelConstraints(
        context_window=1000000, max_input_tokens=1000000, max_output_tokens=4096
    )

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=204.3, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=74.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.0005, output_price_per_1k=0.001)),
        release_date=datetime(2024, 5, 24),
        data_cutoff_date=datetime(2023, 11, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gemini-1.5-flash",
        litellm="vertex_ai/gemini-1.5-flash-002",
        langfuse="gemini-1.5-flash",
        lmarena="gemini-1.5-flash-002",
        openrouter="google/gemini-1.5-flash:free",
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
            default_factory=lambda: ParameterValue[int | NotGiven](variant=ParameterVariant.FIXED, value=40)
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
