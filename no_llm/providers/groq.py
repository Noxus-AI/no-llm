from typing import Literal

from pydantic import Field

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar


class GroqProvider(Provider):
    """Groq provider configuration"""

    type: Literal["groq"] = "groq"
    name: str = "Groq"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$GROQ_API_KEY"),
        description="Name of environment variable containing API key",
    )
