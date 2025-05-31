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
from no_llm.providers import DeepseekProvider, OpenRouterProvider, Providers


class DeepseekChatConfiguration(ModelConfiguration):
    """Configuration for DeepSeek Chat model"""

    identity: ModelIdentity = ModelIdentity(
        id="deepseek-chat",
        name="DeepSeek V3",
        version="2024.02",
        description="Advanced multimodal model from DeepSeek with strong performance across a wide range of tasks.",
        creator="DeepSeek",
    )

    providers: Sequence[Providers] = [OpenRouterProvider(), DeepseekProvider()]

    mode: ModelMode = ModelMode.CHAT

    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.TOOLS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
    }

    constraints: ModelConstraints = ModelConstraints(max_input_tokens=65536, max_output_tokens=8192)

    properties: ModelProperties | None = ModelProperties(
        speed=SpeedProperties(score=26.0, label="Average", description="Average (1-3 seconds)"),
        quality=QualityProperties(score=80.0, label="High", description="High Quality"),
    )

    metadata: ModelMetadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(token_prices=TokenPrices(input_price_per_1k=0.00027, output_price_per_1k=0.0011)),
        release_date=datetime(2024, 12, 1),
        data_cutoff_date=datetime(2024, 10, 1),
    )

    integration_aliases: IntegrationAliases | None = IntegrationAliases(
        pydantic_ai="deepseek-chat",
        litellm="deepseek/deepseek-chat",
        langfuse="deepseek-chat",
        lmarena="deepseek-v3",
        openrouter="deepseek/deepseek-chat-v3-0324:free",
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
