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
from no_llm.providers import FireworksProvider, GroqProvider, OpenRouterProvider, Providers, TogetherProvider


class Llama3370BConfiguration(ModelConfiguration):
    """Configuration for LLama 3.3 70B model"""

    identity: ModelIdentity = ModelIdentity(
        id="llama-3.3-70b",
        name="LLama 3.3 70B",
        version="2024.02",
        description="Newest and most advanced model from Meta",
        creator="Meta",
    )

    providers: Sequence[Providers] = [FireworksProvider(), TogetherProvider(), GroqProvider(), OpenRouterProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=128000, max_output_tokens=4096)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=101.0, label="High", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=41.0, label="Very High", description="Very High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00059, output_price_per_1k=0.00099)),
        release_date=datetime(2024, 10, 1),
        data_cutoff_date=datetime(2024, 8, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="llama-3.3-70b-specdec",
        litellm="groq/llama-3.3-70b-specdec",
        langfuse="llama-3.3-70b-specdec",
        lmarena="llama-3.3-70b-instruct:free",
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
