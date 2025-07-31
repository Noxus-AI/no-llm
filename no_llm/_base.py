from pydantic import BaseModel


class BaseResource(BaseModel):
    is_active: bool = True

    @property
    def is_valid(self) -> bool:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)
