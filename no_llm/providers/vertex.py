from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar

if TYPE_CHECKING:
    from collections.abc import Iterator


class VertexProvider(Provider):
    """Google Vertex AI provider configuration"""

    type: Literal["vertex"] = "vertex"
    name: str = "Vertex AI"
    project_id: EnvVar[str] = Field(default_factory=lambda: EnvVar[str]("$VERTEX_PROJECT_ID"))
    locations: list[str] = Field(default=["us-central1", "europe-west1"])
    _value: str | None = PrivateAttr(default=None)

    def iter(self) -> Iterator[Provider]:
        if not self.has_valid_env():
            return

        for location in self.locations:
            provider = self.model_copy()
            provider._value = location  # noqa: SLF001
            yield provider

    @property
    def current(self) -> str:
        """Get current value, defaulting to first location if not set"""
        return self._value or self.locations[0]

    def reset_variants(self) -> None:
        self._value = None
