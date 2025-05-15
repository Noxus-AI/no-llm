import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ValidationMode(Enum):
    ERROR = "error"
    WARN = "warn"
    CLAMP = "clamp"


class Settings(BaseModel):
    validation_mode: ValidationMode = Field(
        default=ValidationMode(os.getenv("NO_LLM_VALIDATION_MODE", ValidationMode.CLAMP.value)),
        description="Validation mode for model configurations",
    )
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the no_llm library",
    )


settings = Settings()
