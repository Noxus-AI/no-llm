from __future__ import annotations

from pydantic import BaseModel


class IntegrationAliases(BaseModel):
    litellm: str | None = None
    langchain: str | None = None
    llama_index: str | None = None
    pydantic_ai: str | None = None
    langfuse: str | None = None
    lmarena: str | None = None
    openrouter: str | None = None
    fireworks: str | None = None
    together: str | None = None
