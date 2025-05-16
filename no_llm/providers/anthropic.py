from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_ai.providers.anthropic import (
    AnthropicProvider as PydanticAnthropicProvider,
)

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar


class AnthropicProvider(Provider):
    """Anthropic provider configuration"""

    type: Literal["anthropic"] = "anthropic"
    name: str = "Anthropic"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$ANTHROPIC_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: EnvVar[str] | None = Field(default=None, description="Optional base URL override")

    def to_pydantic(self) -> PydanticAnthropicProvider:
        return PydanticAnthropicProvider(
            api_key=str(self.api_key),
        )
