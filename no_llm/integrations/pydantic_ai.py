from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

try:
    from anthropic import AsyncAnthropicVertex
    from mistralai_gcp import MistralGoogleCloud
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        StreamedResponse,
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
    from pydantic_ai.settings import ModelSettings as PydanticModelSettings
except ImportError as _import_error:
    msg = (
        "Please install pydantic-ai to use the Pydantic AI integration, "
        'you can use the `pydantic-ai` optional group — `pip install "no_llm[pydantic-ai]"`'
    )
    raise ImportError(msg) from _import_error

from loguru import logger

from no_llm.config.enums import ModelMode
from no_llm.config.model import ModelConfiguration
from no_llm.integrations._utils import pydantic_mistral_gcp_patch
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import usage
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
    )

ToolName = str


@dataclass
class NoLLMModel(Model):
    _pydantic_model: Model | None = field(default=None, repr=False)
    _current_model_config: ModelConfiguration | None = field(default=None, repr=False)

    def __init__(
        self,
        default_model: ModelConfiguration,
        *fallback_models: ModelConfiguration,
    ):
        self._pydantic_model = None
        self._current_model_config = None
        self.models: list[tuple[Model, ModelConfiguration]] = self._get_pydantic_models(
            [default_model, *fallback_models]
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return "no_llm"

    @property
    def system(self) -> str | None:  # type: ignore
        """The system / model provider, ex: openai."""
        return "no_llm"

    def _get_pydantic_models(
        self,
        model_cfgs: list[ModelConfiguration],
    ) -> list[tuple[Model, ModelConfiguration]]:
        """Get the appropriate pydantic-ai model based on no_llm configuration."""
        models: list[tuple[Model, ModelConfiguration]] = []

        for model_cfg in model_cfgs:
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
                            provider=PydanticOpenAIProvider(
                                api_key=str(provider.api_key), base_url=str(provider.base_url)
                            ),
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
                            provider=PydanticOpenAIProvider(
                                api_key=str(provider.api_key), base_url=str(provider.base_url)
                            ),
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

    def _get_model_settings(
        self,
        model: ModelConfiguration,
        user_settings: PydanticModelSettings | None = None,
    ) -> PydanticModelSettings:
        """Get merged model settings from no_llm config and user settings."""
        if user_settings is not None:
            model.parameters.set_parameters(**user_settings)
        return PydanticModelSettings(**model.parameters.get_model_parameters().get_parameters())  # type: ignore

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: PydanticModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        last_error = None
        for pyd_model, model in self.models:
            try:
                merged_settings = self._get_model_settings(model, model_settings)
                customized_request_parameters = pyd_model.customize_request_parameters(model_request_parameters)
                return await pyd_model.request(messages, merged_settings, customized_request_parameters)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(f"Model {model.identity.id} failed, trying next fallback. Error: {e}")
                continue

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg) from last_error

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: PydanticModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        last_error = None
        for pyd_model, model in self.models:
            try:
                merged_settings = self._get_model_settings(model, model_settings)
                customized_request_parameters = pyd_model.customize_request_parameters(model_request_parameters)
                async with pyd_model.request_stream(
                    messages, merged_settings, customized_request_parameters
                ) as response:
                    yield response
                    return
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(f"Model {model.identity.id} failed, trying next fallback. Error: {e}")
                continue

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg)
