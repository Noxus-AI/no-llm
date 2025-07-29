from pydantic import BaseModel, Field


class SpeedProperties(BaseModel):
    score: float = Field(description="Speed score in tokens per second")
    label: str
    description: str


class QualityProperties(BaseModel):
    score: float = Field(description="Quality score on a 0-100 scale")
    label: str
    description: str


class ModelProperties(BaseModel):
    """Core model quality properties"""

    speed: SpeedProperties = Field(description="Speed properties")
    quality: QualityProperties = Field(description="Quality properties")
