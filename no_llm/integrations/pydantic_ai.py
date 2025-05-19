from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from no_llm.config.parameters import ModelParameters

try:
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        StreamedResponse,
    )
except ImportError as _import_error:
    msg = (
        "Please install pydantic-ai to use the Pydantic AI integration, "
        'you can use the `pydantic-ai` optional group — `pip install "no_llm[pydantic-ai]"`'
    )
    raise ImportError(msg) from _import_error

from loguru import logger

from no_llm.config.model import ModelConfiguration
from no_llm.integrations._utils import _get_pydantic_model

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
    )
    from pydantic_ai.settings import ModelSettings as PydanticModelSettings

ToolName = str
ModelPair = tuple[Model, ModelConfiguration]


@dataclass
class NoLLMModel(Model):
    def __init__(
        self,
        default_model: ModelConfiguration,
        *fallback_models: ModelConfiguration,
    ):
        self.models: list[ModelPair] = self._get_pydantic_models([default_model, *fallback_models])
        self._current_model: ModelPair = self.models[0]

    @property
    def current_model(self) -> Model:
        return self._current_model[0]

    @property
    def current_model_config(self) -> ModelConfiguration:
        return self._current_model[1]

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
            models.extend(_get_pydantic_model(model_cfg))
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
        new_model = model.model_copy(deep=True)
        if user_settings is not None:
            new_model.set_parameters(ModelParameters.from_pydantic(user_settings))
        return new_model.to_pydantic_settings()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: PydanticModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        last_error = None
        for pyd_model, model in self.models:
            try:
                self._current_model = (pyd_model, model)
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
                self._current_model = (pyd_model, model)
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
