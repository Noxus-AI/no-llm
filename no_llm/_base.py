from pydantic import BaseModel


class BaseResource(BaseModel):
    is_active: bool = True

    @property
    def is_valid(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
