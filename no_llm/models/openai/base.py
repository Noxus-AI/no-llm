from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models.openai import OpenAIModelSettings

from no_llm import ModelCapability
from no_llm.config import (
    ModelConfiguration,
)
from no_llm.config.parameters import NOT_GIVEN

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings


class OpenaiBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> ModelSettings:
        base = super().to_pydantic_settings()
        reasoning_effort = base.pop("reasoning_effort", "off")
        if ModelCapability.REASONING in self.capabilities and reasoning_effort not in [
            None,
            "off",
            NOT_GIVEN,
        ]:
            return OpenAIModelSettings(
                **base,
                openai_reasoning_effort=reasoning_effort,  # type: ignore
            )  # type: ignore
        return base
