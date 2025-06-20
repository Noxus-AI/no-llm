from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from no_llm import ModelCapability
from no_llm.config import (
    ModelConfiguration,
)
from no_llm.config.parameters import NOT_GIVEN

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class OpenaiBaseConfiguration(ModelConfiguration):
    def to_pydantic_model(self) -> Model:
        return super().to_pydantic_model()

    def to_pydantic_settings(self) -> OpenAIResponsesModelSettings:
        base = super().to_pydantic_settings()
        reasoning_effort = cast(str, base.pop("reasoning_effort", "off"))
        # nbase = cast(dict, {f"openai_{k}": v for k, v in base.items()})
        if ModelCapability.REASONING in self.capabilities and reasoning_effort not in [
            None,
            "off",
            NOT_GIVEN,
        ]:
            return OpenAIResponsesModelSettings(
                **base,
                openai_reasoning_effort=reasoning_effort,  # type: ignore
                openai_reasoning_summary="detailed",
            )  # type: ignore
        return OpenAIResponsesModelSettings(**base)
