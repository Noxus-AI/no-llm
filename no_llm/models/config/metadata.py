from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from no_llm.errors import InvalidPricingConfigError


class PrivacyLevel(str, Enum):
    BASIC = "basic"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FEDRAMP = "fedramp"
    SOC2 = "soc2"


class TokenPrices(BaseModel):
    input_price_per_1k: float = Field(ge=0, description="Price per 1k input tokens")
    output_price_per_1k: float = Field(ge=0, description="Price per 1k output tokens")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        input_cost = input_tokens * self.input_price_per_1k / 1000
        output_cost = output_tokens * self.output_price_per_1k / 1000
        return input_cost, output_cost


class CharacterPrices(BaseModel):
    input_price_per_1k: float = Field(ge=0, description="Price per 1k input characters")
    output_price_per_1k: float = Field(ge=0, description="Price per 1k output characters")

    def calculate_cost(self, input_chars: int, output_chars: int) -> tuple[float, float]:
        input_cost = input_chars * self.input_price_per_1k / 1000
        output_cost = output_chars * self.output_price_per_1k / 1000
        return input_cost, output_cost


class ModelPricing(BaseModel):
    token_prices: TokenPrices | None = None
    character_prices: CharacterPrices | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_pricing_type(cls, data: dict) -> dict:
        if isinstance(data, dict):
            has_token_prices = "token_prices" in data and data["token_prices"] is not None
            has_char_prices = "character_prices" in data and data["character_prices"] is not None
            if not has_token_prices and not has_char_prices:
                raise InvalidPricingConfigError
        return data

    def calculate_cost(self, input_size: int, output_size: int) -> tuple[float, float]:
        if self.token_prices is not None:
            return self.token_prices.calculate_cost(input_size, output_size)
        if self.character_prices is not None:
            return self.character_prices.calculate_cost(input_size, output_size)
        raise InvalidPricingConfigError


class ModelMetadata(BaseModel):
    privacy_level: list[PrivacyLevel] = Field(description="Privacy level of the model")
    pricing: ModelPricing = Field(description="Pricing information")
    release_date: datetime = Field(description="Model release date")
    data_cutoff_date: datetime | None = Field(default=None, description="Training data cutoff date")
