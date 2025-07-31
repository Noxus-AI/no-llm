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
