from __future__ import annotations

from typing import TYPE_CHECKING, cast

from google.genai.types import ThinkingConfigDict
from pydantic_ai.models.google import GoogleModelSettings

from no_llm.models.config import ModelCapability
from no_llm.models.config import (
    ModelConfiguration,
)
from no_llm.models.config.parameters import NOT_GIVEN

if TYPE_CHECKING:
    from pydantic_ai.models import Model

THINKING_BUDGET = {
    "low": 512,
    "medium": 1024,
    "high": 4096,
}


class GeminiBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> GoogleModelSettings:
        base = super().to_pydantic_settings()
        # nbase = cast(dict, {f"google_{k}": v for k, v in base.items()})
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        if ModelCapability.REASONING not in self.capabilities:
            return GoogleModelSettings(**base)

        elif reasoning_effort in ["off", NOT_GIVEN]:
            include_thoughts = False
            thinking_budget = 0
        else:
            include_thoughts = True
            thinking_budget = THINKING_BUDGET[reasoning_effort]  # type: ignore
        return GoogleModelSettings(
            **base,
            google_thinking_config=ThinkingConfigDict(
                include_thoughts=include_thoughts, thinking_budget=thinking_budget
            ),
        )
