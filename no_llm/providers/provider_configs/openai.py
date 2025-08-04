from __future__ import annotations

from typing import Literal

import httpx
from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.providers.config import ProviderConfiguration
from no_llm.providers.env_var import EnvVar


class OpenAIProvider(ProviderConfiguration):
    """OpenAI provider configuration"""

    type: Literal["openai"] = "openai"  # type: ignore
    id: str = "openai"
    name: str = "OpenAI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$OPENAI_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.openai.com/v1/", description="Optional base URL override")

    def test(self) -> bool:
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception:
            return False

    def to_pydantic(self) -> PydanticOpenAIProvider:
        return PydanticOpenAIProvider(
            api_key=str(self.api_key),
            base_url=str(self.base_url),
        )
