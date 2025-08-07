from __future__ import annotations

from typing import Literal

import httpx
from loguru import logger
from pydantic import Field
from pydantic_ai.providers.google import GoogleProvider as PydanticGoogleProvider

from no_llm.providers.config import ProviderConfiguration
from no_llm.providers.env_var import EnvVar


class GeminiProvider(ProviderConfiguration):
    """Gemini provider configuration"""

    type: Literal["gemini"] = "gemini"  # type: ignore
    id: str = "gemini"
    name: str = "Gemini"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$GEMINI_API_KEY"),
        description="Name of environment variable containing API key",
    )

    def test(self) -> bool:
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"Authorization": f"Bearer {self.api_key!s}"},
                )
                return response.status_code == 200
        except Exception as e:
            logger.opt(exception=e).error(f"Failed to test connectivity to {self.__class__.__name__}")
            return False

    def to_pydantic(self) -> PydanticGoogleProvider:
        return PydanticGoogleProvider(
            api_key=str(self.api_key),
        )
