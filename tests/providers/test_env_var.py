from __future__ import annotations

import pytest
from no_llm.providers import EnvVar
from pydantic import BaseModel


class TestModel(BaseModel):
    api_key: EnvVar[str] = EnvVar('$TEST_API_KEY')
    optional_key: EnvVar[str] | None = None


def test_env_var_basic():
    env_var = EnvVar[str]('$TEST_VAR')
    assert env_var.var_name == '$TEST_VAR'
    assert repr(env_var) == '$TEST_VAR'


def test_env_var_validation():
    # Should raise error if var name doesn't start with $
    with pytest.raises(ValueError) as exc_info:
        EnvVar[str]('TEST_VAR')
    assert 'Environment variable name must start with $' in str(exc_info.value)


def test_env_var_with_pydantic():
    model = TestModel(api_key=EnvVar[str]('$TEST_KEY'))
    assert isinstance(model.api_key, EnvVar)
    assert model.api_key.var_name == '$TEST_KEY'


def test_env_var_serialization():
    model = TestModel(api_key=EnvVar[str]('$TEST_KEY'))
    serialized = model.model_dump()
    assert serialized['api_key'] == '$TEST_KEY'



def test_env_var_optional():
    # Test with None value
    model = TestModel(api_key=EnvVar[str]('$TEST_KEY'), optional_key=None)
    assert model.optional_key is None

    # Test with actual EnvVar
    model = TestModel(api_key=EnvVar[str]('$TEST_KEY'), optional_key=EnvVar[str]('$OPTIONAL_KEY'))
    assert isinstance(model.optional_key, EnvVar)
    assert model.optional_key.var_name == '$OPTIONAL_KEY'


def test_env_var_in_provider():
    provider = TestModel(
        api_key=EnvVar[str]('$TEST_KEY')
    )
    assert isinstance(provider.api_key, EnvVar)
    assert provider.api_key.var_name == '$TEST_KEY'

    # Test serialization
    serialized = provider.model_dump()
    assert serialized['api_key'] == '$TEST_KEY'


def test_env_var_value_resolution(monkeypatch):
    monkeypatch.setenv('TEST_KEY', 'test_value')
    env_var = EnvVar[str]('$TEST_KEY')
    assert env_var.__get__(None, None) == 'test_value'


def test_env_var_missing_value(monkeypatch):
    env_var = EnvVar[str]('$MISSING_KEY')
    assert env_var.__get__(None, None) == '$MISSING_KEY'


def test_env_var_type_hints():
    # Test with different type hints
    str_var = EnvVar[str]('$STR_VAR')
    int_var = EnvVar[int]('$INT_VAR')
    bool_var = EnvVar[bool]('$BOOL_VAR')

    assert str_var.var_name == '$STR_VAR'
    assert int_var.var_name == '$INT_VAR'
    assert bool_var.var_name == '$BOOL_VAR'


def test_env_var_comparison():
    var1 = EnvVar[str]('$TEST_VAR')
    var2 = EnvVar[str]('$TEST_VAR')
    var3 = EnvVar[str]('$OTHER_VAR')

    # Test string representation equality
    assert str(var1) == '$TEST_VAR'
    assert repr(var1) == '$TEST_VAR'

    # Test var_name comparison
    assert var1.var_name == var2.var_name
    assert var1.var_name != var3.var_name


def test_env_var_provider_serialization():
    provider = TestModel(
        api_key=EnvVar[str]('$TEST_KEY'),
    )
    serialized = provider.model_dump()
    assert serialized['api_key'] == '$TEST_KEY'

def test_env_var_is_valid(monkeypatch):
    # Test with unset env var
    monkeypatch.delenv('TEST_KEY', raising=False)
    env_var = EnvVar[str]('$TEST_KEY')
    assert not env_var.is_valid()

    # Test with set env var
    monkeypatch.setenv('TEST_KEY', 'test_value')
    env_var = EnvVar[str]('$TEST_KEY')
    assert env_var.is_valid()


def test_env_var_value_refresh(monkeypatch):
    # Set initial environment variable
    monkeypatch.setenv('TEST_CACHE', 'initial_value')
    env_var = EnvVar[str]('$TEST_CACHE')
    assert env_var.__get__(None, None) == 'initial_value'

    # Change environment variable - should get new value
    monkeypatch.setenv('TEST_CACHE', 'new_value')
    assert env_var.__get__(None, None) == 'new_value'

    # Remove environment variable - should get default
    monkeypatch.delenv('TEST_CACHE')
    assert env_var.__get__(None, None) == '$TEST_CACHE'

