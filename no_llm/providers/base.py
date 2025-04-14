from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr, model_serializer

from no_llm.providers.env_var import EnvVar

if TYPE_CHECKING:
    from collections.abc import Iterator


class ParameterMapping(BaseModel):
    name: str | None = Field(None, description="Provider-specific parameter name")
    supported: bool = Field(default=True, description="Whether parameter is supported by provider")


class Provider(BaseModel):
    """Base provider configuration"""

    # type: str = Field(description="Provider type")
    name: str = Field(description="Provider name for display")
    parameter_mappings: dict[str, ParameterMapping] = Field(
        default_factory=dict, description="Mapping of standard parameters to provider-specific parameters"
    )
    _iterator_index: int = PrivateAttr(default=0)

    def iter(self) -> Iterator[Provider]:
        """Default implementation yields just the provider itself"""
        if self.has_valid_env():
            yield self

    def has_valid_env(self) -> bool:
        """Check if all required environment variables are set"""
        for field_name, field in self.model_fields.items():
            if field.annotation == EnvVar[str] and not getattr(self, field_name).is_valid():
                return False
        return True

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {}
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if field_name == "parameter_mappings":
                continue
            if isinstance(value, EnvVar):
                result[field_name] = value.__get__(None, None)
            else:
                result[field_name] = value
        return result

    def map_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Maps standard parameters to provider-specific parameters."""
        result = {}
        for param_name, value in params.items():
            mapping = self.parameter_mappings.get(param_name)
            if mapping:
                if mapping.supported:
                    result[mapping.name] = value
            else:
                result[param_name] = value
        return result  # type: ignore

    def reset_iterator(self) -> None:
        """Reset iteration state"""
        self._iterator_index = 0
