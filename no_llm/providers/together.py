from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class TogetherProvider(OpenAIProvider):
    """Together provider configuration"""

    type: Literal["together"] = "together"  # type: ignore
    name: str = "TogetherAI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$TOGETHER_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.together.xyz/v1", description="Base URL for Together API")

    def to_pydantic(self) -> PydanticOpenAIProvider:
        return PydanticOpenAIProvider(
            api_key=str(self.api_key),
            base_url=str(self.base_url),
        )
