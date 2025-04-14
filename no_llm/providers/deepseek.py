from __future__ import annotations

from typing import Literal

from pydantic import Field

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class DeepseekProvider(OpenAIProvider):
    """Deepseek provider configuration"""

    type: Literal["deepseek"] = "deepseek"  # type: ignore
    name: str = "DeepSeek"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$DEEPSEEK_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.deepseek.com", description="Base URL for Deepseek API")
