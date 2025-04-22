import sys
from typing import cast

from anthropic import AsyncAnthropicVertex
from loguru import logger
from mistralai_gcp import MistralGoogleCloud
from pydantic_ai.models import (
    Model,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider
from pydantic_ai.providers.azure import AzureProvider as PydanticAzureProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider as PydanticVertexProvider
from pydantic_ai.providers.google_vertex import VertexAiRegion
from pydantic_ai.providers.groq import GroqProvider as PydanticGroqProvider
from pydantic_ai.providers.mistral import MistralProvider as PydanticMistralProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.config.enums import ModelMode
from no_llm.config.model import ModelConfiguration
from no_llm.providers import (
    AnthropicProvider,
    AzureProvider,
    DeepseekProvider,
    FireworksProvider,
    GrokProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
    TogetherProvider,
    VertexProvider,
)

from .. import ModelConfiguration


def pydantic_mistral_gcp_patch():
    from mistralai_gcp import (
        CompletionChunk as MistralCompletionChunk,
    )
    from mistralai_gcp import (
        Content as MistralContent,
    )
    from mistralai_gcp import (
        ContentChunk as MistralContentChunk,
    )
    from mistralai_gcp import (
        FunctionCall as MistralFunctionCall,
    )
    from mistralai_gcp import (
        OptionalNullable as MistralOptionalNullable,
    )
    from mistralai_gcp import (
        TextChunk as MistralTextChunk,
    )
    from mistralai_gcp import (
        ToolChoiceEnum as MistralToolChoiceEnum,
    )
    from mistralai_gcp.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
    )
    from mistralai_gcp.models import (
        CompletionEvent as MistralCompletionEvent,
    )
    from mistralai_gcp.models import (
        Messages as MistralMessages,
    )
    from mistralai_gcp.models import (
        Tool as MistralTool,
    )
    from mistralai_gcp.models import (
        ToolCall as MistralToolCall,
    )
    from mistralai_gcp.models.assistantmessage import AssistantMessage as MistralAssistantMessage
    from mistralai_gcp.models.function import Function as MistralFunction
    from mistralai_gcp.models.systemmessage import SystemMessage as MistralSystemMessage
    from mistralai_gcp.models.toolmessage import ToolMessage as MistralToolMessage
    from mistralai_gcp.models.usermessage import UserMessage as MistralUserMessage
    from mistralai_gcp.types.basemodel import Unset as MistralUnset
    from mistralai_gcp.utils.eventstreaming import EventStreamAsync as MistralEventStreamAsync

    # from mistralai_gcp.models.imageurl import (
    #     ImageURL as MistralImageURL,
    #     ImageURLChunk as MistralImageURLChunk,
    # )

    sys.modules["pydantic_ai.models.mistral"].MistralUserMessage = MistralUserMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralSystemMessage = MistralSystemMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralAssistantMessage = MistralAssistantMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunction = MistralFunction  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolMessage = MistralToolMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralChatCompletionResponse = MistralChatCompletionResponse  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionEvent = MistralCompletionEvent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralMessages = MistralMessages  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTool = MistralTool  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolCall = MistralToolCall  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralUnset = MistralUnset  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralEventStreamAsync = MistralEventStreamAsync  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralOptionalNullable = MistralOptionalNullable  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTextChunk = MistralTextChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolChoiceEnum = MistralToolChoiceEnum  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionChunk = MistralCompletionChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContent = MistralContent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContentChunk = MistralContentChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunctionCall = MistralFunctionCall  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURL = MistralImageURL  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURLChunk = MistralImageURLChunk  # type: ignore


def _get_pydantic_model(
    model_cfg: ModelConfiguration,
) -> list[tuple[Model, ModelConfiguration]]:
    """Get the appropriate pydantic-ai model based on no_llm configuration."""
    models: list[tuple[Model, ModelConfiguration]] = []

    if model_cfg.integration_aliases is None:
        msg = "Model must have integration aliases. It is required for pydantic-ai integration."
        raise TypeError(msg)
    if model_cfg.integration_aliases.pydantic_ai is None:
        msg = "Model must have a pydantic-ai integration alias. It is required for pydantic-ai integration."
        raise TypeError(msg)
    if model_cfg.mode != ModelMode.CHAT:
        msg = f"Model {model_cfg.identity.id} must be a chat model"
        raise TypeError(msg)
    pyd_model: Model | None = None
    for provider in model_cfg.iter():
        try:
            if isinstance(provider, VertexProvider):
                if "mistral" in model_cfg.identity.id:
                    pydantic_mistral_gcp_patch()
                    pyd_model = MistralModel(
                        model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                        provider=PydanticMistralProvider(
                            mistral_client=MistralGoogleCloud(  # type: ignore
                                project_id=str(provider.project_id), region=str(provider.current)
                            ),
                        ),
                    )
                elif "claude" in model_cfg.identity.id:
                    pyd_model = AnthropicModel(
                        model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                        provider=PydanticAnthropicProvider(
                            anthropic_client=AsyncAnthropicVertex(  # type: ignore
                                project_id=str(provider.project_id), region=str(provider.current)
                            ),
                        ),
                    )
                elif "gemini" in model_cfg.identity.id:
                    pyd_model = GeminiModel(
                        model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                        provider=PydanticVertexProvider(
                            region=cast(VertexAiRegion, provider.current),
                            project_id=str(provider.project_id),
                        ),
                    )
            elif isinstance(provider, AnthropicProvider):
                pyd_model = AnthropicModel(
                    model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                    provider=PydanticAnthropicProvider(
                        api_key=str(provider.api_key),
                    ),
                )
            elif isinstance(provider, MistralProvider):
                pyd_model = MistralModel(
                    model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                    provider=PydanticMistralProvider(api_key=str(provider.api_key)),
                )
            elif isinstance(provider, GroqProvider):
                pyd_model = GroqModel(
                    model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                    provider=PydanticGroqProvider(api_key=str(provider.api_key)),
                )
            elif isinstance(provider, OpenRouterProvider):
                pyd_model = OpenAIModel(
                    model_name=model_cfg.integration_aliases.openrouter or model_cfg.identity.id,
                    provider=PydanticOpenAIProvider(api_key=str(provider.api_key), base_url=str(provider.base_url)),
                )
            elif isinstance(provider, AzureProvider):
                pyd_model = OpenAIModel(
                    model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                    provider=PydanticAzureProvider(
                        api_key=str(provider.api_key), azure_endpoint=str(provider.base_url)
                    ),
                )
            elif isinstance(
                provider,
                OpenAIProvider
                | DeepseekProvider
                | PerplexityProvider
                | FireworksProvider
                | TogetherProvider
                | GrokProvider,
            ):
                pyd_model = OpenAIModel(
                    model_name=model_cfg.integration_aliases.pydantic_ai or model_cfg.identity.id,
                    provider=PydanticOpenAIProvider(api_key=str(provider.api_key), base_url=str(provider.base_url)),
                )
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=e).warning(f"Failed to create model for provider {type(provider).__name__}")
            continue
        if pyd_model is not None:
            models.append((pyd_model, model_cfg))

    if not models:
        msg = "Couldn't build any models for pydantic-ai integration"
        raise RuntimeError(msg)
    return models
