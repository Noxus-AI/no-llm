from __future__ import annotations

from no_llm.providers import BedrockProvider, EnvVar


def test_bedrock_provider_iter():
    provider = BedrockProvider(
        region=EnvVar[str]('$BEDROCK_REGION'),
        locations=['us-east-1', 'us-east-2']
    )

    # Test iteration
    providers = list(provider.iter())
    assert len(providers) == 0


def test_bedrock_provider_current():
    provider = BedrockProvider(
        region=EnvVar[str]('$BEDROCK_REGION'),
        locations=['us-east-1', 'us-east-2']
    )

    # Test default current (first location)
    assert provider.current == 'us-east-1'

    # Test after setting _value
    provider._value = 'us-east-2'
    assert provider.current == 'us-east-2'


def test_bedrock_provider_reset_variants():
    provider = BedrockProvider(
        region=EnvVar[str]('$BEDROCK_REGION'),
        locations=['us-east-1', 'us-east-2']
    )

    provider._value = 'us-east-2'
    provider.reset_variants()
    assert provider._value is None
    assert provider.current == 'us-east-1'  # Should return first location after reset
