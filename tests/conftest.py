import pytest
from pathlib import Path
from vcr import VCR
from typing import Any
from no_llm.models.registry import ModelRegistry
from no_llm.settings import ValidationMode, settings

@pytest.fixture(scope='module')
def vcr_cassette_dir(request):
    """Configure VCR cassette directory"""
    return str(Path(__file__).parent / "vcr_cassettes")

import brotli
def decompress_response(response):
    if response['headers'].get('Content-Encoding') == ['br']:
        response['body']['string'] = brotli.decompress(response['body']['string'])
        del response['headers']['Content-Encoding']
    return response

@pytest.fixture(scope='module')
def vcr_config(vcr_cassette_dir: str):
    """VCR configuration"""
    return {
        'ignore_localhost': True,
        "record_mode": "once",
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
        'before_record_response': decompress_response,
    }

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope='session')
def builtin_model_registry():
    registry = ModelRegistry()
    yield registry

@pytest.fixture(scope='session')
def no_llm_error_settings():
    original_validation_mode = settings.validation_mode
    settings.validation_mode = ValidationMode.ERROR
    yield settings
    settings.validation_mode = original_validation_mode

@pytest.fixture(scope='session')
def no_llm_warn_settings():
    original_validation_mode = settings.validation_mode
    settings.validation_mode = ValidationMode.WARN
    yield settings
    settings.validation_mode = original_validation_mode

@pytest.fixture(scope='session')
def no_llm_clamp_settings():
    original_validation_mode = settings.validation_mode
    settings.validation_mode = ValidationMode.CLAMP
    yield settings
    settings.validation_mode = original_validation_mode
