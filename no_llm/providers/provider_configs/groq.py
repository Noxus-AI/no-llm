from typing import Literal

import httpx
from loguru import logger
from pydantic import Field
from pydantic_ai.providers.groq import GroqProvider as PydanticGroqProvider

from no_llm.providers.config import ProviderConfiguration
from no_llm.providers.env_var import EnvVar


class GroqProvider(ProviderConfiguration):
    """Groq provider configuration"""

    type: Literal["groq"] = "groq"  # type: ignore
    id: str = "groq"
    name: str = "Groq"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$GROQ_API_KEY"),
        description="Name of environment variable containing API key",
    )

    def test(self) -> bool:
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key!s}"},
                )
                return response.status_code == 200
        except Exception as e:
            logger.opt(exception=e).error(f"Failed to test connectivity to {self.__class__.__name__}")
            return False

    def to_pydantic(self) -> PydanticGroqProvider:
        return PydanticGroqProvider(
            api_key=str(self.api_key),
        )
