from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from no_llm.models.config.metadata import (
    CharacterPrices,
    ModelMetadata,
    ModelPricing,
    PrivacyLevel,
    TokenPrices,
)
from no_llm.errors import InvalidPricingConfigError


def test_token_prices_calculation():
    prices = TokenPrices(input_price_per_1k=0.5, output_price_per_1k=1.0)
    
    # Test with round numbers
    input_cost, output_cost = prices.calculate_cost(1000, 2000)
    assert input_cost == 0.5  # 1000 tokens * 0.5/1000
    assert output_cost == 2.0  # 2000 tokens * 1.0/1000
    
    # Test with fractional numbers
    input_cost, output_cost = prices.calculate_cost(500, 1500)
    assert input_cost == 0.25  # 500 tokens * 0.5/1000
    assert output_cost == 1.5  # 1500 tokens * 1.0/1000


def test_character_prices_calculation():
    prices = CharacterPrices(input_price_per_1k=0.1, output_price_per_1k=0.2)
    
    # Test with round numbers
    input_cost, output_cost = prices.calculate_cost(1000, 2000)
    assert input_cost == 0.1  # 1000 chars * 0.1/1000
    assert output_cost == 0.4  # 2000 chars * 0.2/1000
    
    # Test with fractional numbers
    input_cost, output_cost = prices.calculate_cost(500, 1500)
    assert input_cost == 0.05  # 500 chars * 0.1/1000
    assert output_cost == 0.3  # 1500 chars * 0.2/1000


def test_model_pricing_validation():
    # Test that pricing must have either token or character prices
    with pytest.raises(InvalidPricingConfigError):
        ModelPricing(token_prices=None, character_prices=None)
    
    # Test valid token pricing
    token_pricing = ModelPricing(
        token_prices=TokenPrices(input_price_per_1k=0.5, output_price_per_1k=1.0),
        character_prices=None
    )
    assert token_pricing.token_prices is not None
    
    # Test valid character pricing
    char_pricing = ModelPricing(
        token_prices=None,
        character_prices=CharacterPrices(input_price_per_1k=0.1, output_price_per_1k=0.2)
    )
    assert char_pricing.character_prices is not None


def test_model_pricing_calculate_cost():
    # Test token-based pricing
    token_pricing = ModelPricing(
        token_prices=TokenPrices(input_price_per_1k=0.5, output_price_per_1k=1.0)
    )
    input_cost, output_cost = token_pricing.calculate_cost(1000, 2000)
    assert input_cost == 0.5
    assert output_cost == 2.0
    
    # Test character-based pricing
    char_pricing = ModelPricing(
        character_prices=CharacterPrices(input_price_per_1k=0.1, output_price_per_1k=0.2)
    )
    input_cost, output_cost = char_pricing.calculate_cost(1000, 2000)
    assert input_cost == 0.1
    assert output_cost == 0.4
    
    # Test error when neither pricing is set
    with pytest.raises(InvalidPricingConfigError):
        invalid_pricing = ModelPricing(
            token_prices=None,
            character_prices=None
        )


def test_model_metadata():
    # Test valid metadata
    metadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC, PrivacyLevel.GDPR],
        pricing=ModelPricing(
            token_prices=TokenPrices(input_price_per_1k=0.5, output_price_per_1k=1.0)
        ),
        release_date=datetime(2023, 1, 1),
        data_cutoff_date=datetime(2022, 12, 1)
    )
    assert len(metadata.privacy_level) == 2
    assert metadata.data_cutoff_date is not None
    
    # Test optional data_cutoff_date
    metadata = ModelMetadata(
        privacy_level=[PrivacyLevel.BASIC],
        pricing=ModelPricing(
            token_prices=TokenPrices(input_price_per_1k=0.5, output_price_per_1k=1.0)
        ),
        release_date=datetime(2023, 1, 1)
    )
    assert metadata.data_cutoff_date is None


def test_pricing_validation():
    # Test negative prices are not allowed
    with pytest.raises(ValidationError):
        TokenPrices(input_price_per_1k=-0.5, output_price_per_1k=1.0)
    
    with pytest.raises(ValidationError):
        CharacterPrices(input_price_per_1k=0.1, output_price_per_1k=-0.2) 