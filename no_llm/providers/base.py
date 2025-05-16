from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, get_args

from pydantic import BaseModel, Field, model_serializer, model_validator

from no_llm.providers.env_var import EnvVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic_ai.providers import Provider as PydanticProvider


class ParameterMapping(BaseModel):
    name: str | None = Field(None, description="Provider-specific parameter name")
    supported: bool = Field(default=True, description="Whether parameter is supported by provider")


class Provider(BaseModel):
    """Base provider configuration"""

    name: str = Field(description="Provider name for display")
    parameter_mappings: dict[str, ParameterMapping] = Field(
        default_factory=dict,
        description="Mapping of standard parameters to provider-specific parameters",
    )

    def iter(self) -> Iterator[Provider]:
        """Default implementation yields just the provider itself"""
        if self.has_valid_env():
            yield self

    def has_valid_env(self) -> bool:
        """Check if all required environment variables are set"""
        for field_name, field in self.__class__.model_fields.items():
            if field.annotation == EnvVar[str] and not getattr(self, field_name).is_valid():
                return False
        return True

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if field_name == "parameter_mappings":
                continue
            if isinstance(value, EnvVar):
                result[field_name] = value.__get__(None, None)
            else:
                result[field_name] = value
        return result

    @model_validator(mode="before")
    @classmethod
    def convert_env_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        for field_name, field in cls.model_fields.items():
            if field_name not in data:
                continue

            value = data[field_name]
            if not isinstance(value, str) or not value.startswith("$"):
                continue

            if field.annotation and getattr(field.annotation, "__origin__", None) is EnvVar:
                args = get_args(field.annotation)
                if args and args[0] is str:
                    data[field_name] = EnvVar(value)

        return data

    @abstractmethod
    def to_pydantic(self) -> PydanticProvider:
        """Convert provider to Pydantic provider"""
