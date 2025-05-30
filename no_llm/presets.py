from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import assert_never

from loguru import logger
from pydantic import BaseModel, Field

from no_llm import ModelCapability, ModelConfiguration, ModelParameters, ModelRegistry


class ModelPreset(BaseModel):
    models: Sequence[str | ModelPreset] = Field(description="List of model IDs")
    required_capabilities: set[ModelCapability] = Field(
        default=set(), description="Capabilities required for the preset"
    )
    title: str = Field(default="A Model Preset", description="Title of the preset")
    subtitle: str = Field(default="A model preset", description="Subtitle of the preset")
    description: str = Field(default="A model preset", description="Description of the preset")
    parameters: ModelParameters | None = Field(default=None, description="Parameters of the preset")
    data_center_fallback: bool = Field(
        default=True,
        description="Whether to use data center fallback, meaning to change regions of the model when available",
    )
    _current_model: ModelConfiguration | None = None

    def iter(self, registry: ModelRegistry) -> Iterator[ModelConfiguration]:
        for model in self.models:
            if isinstance(model, str):
                model_cfg = registry.get_model(model)
                if len(self.required_capabilities) > 0 and not model_cfg.check_capabilities(self.required_capabilities):
                    logger.warning(
                        f"Model {model} does not have the required capabilities: {self.required_capabilities}. Skipping."
                    )
                    continue
                if self.parameters is not None:
                    model_cfg.set_parameters(self.parameters)

                for provider in model_cfg.iter():
                    providers = provider.iter() if self.data_center_fallback else [provider]
                    for provider_i in providers:
                        copied_cfg = model_cfg.model_copy(deep=True)
                        copied_cfg.providers = (
                            [provider_i] if self.data_center_fallback else [provider]  # type: ignore
                        )
                        self._current_model = copied_cfg
                        yield copied_cfg
            elif isinstance(model, ModelPreset):
                for m in model.iter(registry):
                    if self.parameters is not None:
                        m.set_parameters(self.parameters)

                    yield m
            else:
                assert_never(model)

    def get_current_model(self) -> ModelConfiguration:
        if self._current_model is None:
            msg = "No model selected"
            raise ValueError(msg)
        return self._current_model
