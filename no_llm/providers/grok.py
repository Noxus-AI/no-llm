from __future__ import annotations

from typing import Literal

from pydantic import Field

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class GrokProvider(OpenAIProvider):
    """Grok provider configuration"""

    type: Literal["grok"] = "grok"  # type: ignore
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$GROK_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.x.ai/v1", description="Base URL for Grok API")
