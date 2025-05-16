from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider configuration"""

    type: Literal["openrouter"] = "openrouter"  # type: ignore
    name: str = "OpenRouter"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$OPENROUTER_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API",
    )

    def to_pydantic(self) -> PydanticOpenAIProvider:
        return PydanticOpenAIProvider(
            api_key=str(self.api_key),
            base_url=str(self.base_url),
        )
