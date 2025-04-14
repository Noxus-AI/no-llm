from __future__ import annotations

from typing import Literal

from pydantic import Field

from no_llm.providers.env_var import EnvVar
from no_llm.providers.openai import OpenAIProvider


class FireworksProvider(OpenAIProvider):
    """Fireworks provider configuration"""

    type: Literal["fireworks"] = "fireworks"  # type: ignore
    name: str = "Fireworks"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$FIREWORKS_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(
        default="https://api.fireworks.ai/inference/v1", description="Base URL for Fireworks API"
    )
