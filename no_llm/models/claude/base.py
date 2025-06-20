from __future__ import annotations

from typing import TYPE_CHECKING, cast

from anthropic.types.beta import (
    BetaThinkingConfigDisabledParam,
    BetaThinkingConfigEnabledParam,
)
from pydantic_ai.models.anthropic import AnthropicModelSettings

from no_llm.config import (
    ModelCapability,
    ModelConfiguration,
)
from no_llm.config.parameters import NOT_GIVEN

if TYPE_CHECKING:
    from pydantic_ai.models import Model

THINKING_BUDGET = {
    "low": 1025,
    "medium": 2048,
    "high": 4096,
}


class ClaudeBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> AnthropicModelSettings:
        base = super().to_pydantic_settings()
        # nbase = cast(dict, {f"anthropic_{k}": v for k, v in base.items()})
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        if ModelCapability.REASONING not in self.capabilities:
            return AnthropicModelSettings(**base)
        elif reasoning_effort in ["off", NOT_GIVEN]:
            thinking_config = BetaThinkingConfigDisabledParam(type="disabled")
        else:
            # NOTE: for reasoning temperature needs to be 1
            base["temperature"] = 1.0
            # NOTE: for reasoning max_tokens needs to be at least the thinking budget
            base["max_tokens"] = max(base["max_tokens"], THINKING_BUDGET[reasoning_effort]) + 1  # type: ignore
            thinking_config = BetaThinkingConfigEnabledParam(
                type="enabled", budget_tokens=THINKING_BUDGET[reasoning_effort]
            )
        return AnthropicModelSettings(
            **base,
            anthropic_thinking=thinking_config,
        )
