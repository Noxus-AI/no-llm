from __future__ import annotations

from typing import Literal

from pydantic import Field

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar


class OpenAIProvider(Provider):
    """OpenAI provider configuration"""

    type: Literal["openai"] = "openai"
    name: str = "OpenAI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$OPENAI_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default=None, description="Optional base URL override")
