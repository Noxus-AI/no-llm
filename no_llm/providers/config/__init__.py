from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, model_serializer, model_validator

from no_llm._base import BaseResource
from no_llm.providers.env_var import EnvVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic_ai.providers import Provider as PydanticProvider


class ProviderConfiguration(BaseResource):
    """Base provider configuration"""

    type: Literal["provider"] = "provider"
    id: str = Field(description="Provider ID")
    name: str = Field(description="Provider name for display")

    @property
    def is_valid(self) -> bool:
        return self.has_valid_env()

    def test(self) -> bool:
        msg = "Test method not implemented"
        raise NotImplementedError(msg)

    def iter(self) -> Iterator[ProviderConfiguration]:
        """Default implementation yields just the provider itself"""
        if self.has_valid_env():
            yield self

    def has_valid_env(self) -> bool:
        """Check if all required environment variables are set"""
        # TODO; very fickle implementation, should be improved
        for field_name, field in self.__class__.model_fields.items():
            if (
                field.annotation == EnvVar[str]
                and isinstance(getattr(self, field_name), EnvVar)
                and not getattr(self, field_name).is_valid()
            ):
                return False
        return True

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
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
    def to_pydantic(self, *args, **kwargs) -> PydanticProvider:
        """Convert provider to Pydantic provider"""


Provider = ProviderConfiguration
