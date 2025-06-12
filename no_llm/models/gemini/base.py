from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models.gemini import GeminiModelSettings, ThinkingConfig

from no_llm import ModelCapability
from no_llm.config import (
    ModelConfiguration,
)
from no_llm.config.parameters import NOT_GIVEN

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings

THINKING_BUDGET = {
    "low": 512,
    "medium": 1024,
    "high": 4096,
}


class GeminiBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> ModelSettings:
        base = super().to_pydantic_settings()
        reasoning_effort = base.pop("reasoning_effort", "off")
        if ModelCapability.REASONING not in self.capabilities:
            return base
        elif reasoning_effort in ["off", NOT_GIVEN]:
            include_thoughts = False
            thinking_budget = 0
        else:
            include_thoughts = True
            thinking_budget = THINKING_BUDGET[reasoning_effort]  # type: ignore
        return GeminiModelSettings(
            **base,
            gemini_thinking_config=ThinkingConfig(include_thoughts=include_thoughts, thinking_budget=thinking_budget),
        )
