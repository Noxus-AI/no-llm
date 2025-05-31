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
    SpeedProperties,
    TokenPrices,
)
from no_llm.config.parameters import NOT_GIVEN, NotGiven, RangeValidation
from no_llm.providers import OpenRouterProvider, Providers, VertexProvider


class Gemini20FlashThinkingConfiguration(ModelConfiguration):
    """Configuration for Gemini 2.0 Flash Thinking model"""

    identity: ModelIdentity = ModelIdentity(
        id="gemini-2.0-flash-thinking",
        name="Gemini 2.0 Flash Thinking",
        version="2024.02",
        description="Gemini 2.0 Flash Thinking is a smaller and faster version of Google's best foundation model, performing with respectable quality at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video.",
        creator="Google",
    )

    providers: Sequence[Providers] = [
        VertexProvider(model_family="gemini"),
        OpenRouterProvider(),
    ]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=1048576, max_output_tokens=8192)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=169.0, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=85.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00010, output_price_per_1k=0.0004)),
        release_date=datetime(2024, 12, 1),
        data_cutoff_date=datetime(2024, 12, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="gemini-2.0-flash-thinking-exp-01-21",
        litellm="gemini-2.0-flash-thinking-exp-01-21",
        langfuse="gemini-2.0-flash-thinking-exp-01-21",
        lmarena="gemini-2.0-flash-thinking-exp-01-21",
        openrouter="google/gemini-2.0-flash-thinking-exp:free",
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
