from pathlib import Path
from typing import Annotated, Any, TypeVar, get_args

T = TypeVar("T")


def _get_annotated_union_members(annotated_type: Annotated[Any, ...]) -> list[Any]:
    """Get the members of an annotated union type"""
    annotated_args = get_args(annotated_type)
    if annotated_args:
        # First arg is the Union type, second is the Discriminator
        union_type = annotated_args[0]
        return list(get_args(union_type))
    else:
        return []


def find_yaml_file(base_path: Path, name: str) -> Path:
    for ext in [".yml", ".yaml"]:
        path = base_path / f"{name}{ext}"
        if path.exists():
            return path
    return base_path / f"{name}.yml"


def merge_configs(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merged[key] = merge_configs(base[key], value)
        else:
            merged[key] = value
    return merged
