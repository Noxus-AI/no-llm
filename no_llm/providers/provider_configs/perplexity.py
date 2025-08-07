from __future__ import annotations

from typing import Literal
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.providers.env_var import EnvVar
from no_llm.providers.provider_configs.openai import OpenAIProvider


class PerplexityProvider(OpenAIProvider):
    """Perplexity provider configuration"""

    type: Literal["perplexity"] = "perplexity"  # type: ignore
    id: str = "perplexity"
    name: str = "Perplexity AI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$PERPLEXITY_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: str | None = Field(default="https://api.perplexity.ai/", description="Base URL for Perplexity API")

    def test(self) -> bool:
        try:
            with httpx.Client() as client:
                response = client.get(
                    urljoin(str(self.base_url), "async/chat/completions"),
                    headers={"Authorization": f"Bearer {self.api_key!s}"},
                )
                return response.status_code == 200
        except Exception as e:
            logger.opt(exception=e).error(f"Failed to test connectivity to {self.__class__.__name__}")
            return False

    def to_pydantic(self) -> PydanticOpenAIProvider:
        return PydanticOpenAIProvider(
            api_key=str(self.api_key),
            base_url=str(self.base_url),
        )
