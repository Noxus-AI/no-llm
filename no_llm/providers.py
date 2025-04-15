import os
from typing import Any, Generic, TypeVar

from pydantic_core import CoreSchema, core_schema

T = TypeVar("T")


class EnvVar(Generic[T]):
    def __init__(self, var_name: str) -> None:
        if not var_name.startswith("$"):
            msg = "Environment variable name must start with $"
            raise ValueError(msg)
        self.var_name = var_name
        self._value = None

    def __repr__(self) -> str:
        return self.var_name

    def __str__(self) -> str:
        return self.var_name

    def __get__(self, obj: Any, objtype: Any) -> str:
        if self._value is None:
            env_name = self.var_name[1:]
            self._value = os.environ.get(env_name, self.var_name)
        return self._value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    core_schema.transform_schema(
                        core_schema.str_schema(), lambda x: cls(x) if x.startswith("$") else x
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_schema(
                lambda x: x.var_name if isinstance(x, cls) else x
            ),
        )
