from __future__ import annotations


class ModelNotFoundError(Exception):
    def __init__(self, model_id: str, provider_id: str | None = None):
        self.model_id = model_id
        self.provider_id = provider_id

        if provider_id:
            message = f"Model '{model_id}' not found for provider '{provider_id}'"
        else:
            message = f"Model '{model_id}' not found"

        super().__init__(message)


class ModelRegistrationError(Exception):
    """Base class for model registration errors"""


class DuplicateModelError(ModelRegistrationError):
    def __init__(self, model_id: str):
        self.model_id = model_id
        message = f"Model '{model_id}' is already registered"
        super().__init__(message)


class ConfigurationError(Exception):
    """Base class for configuration errors"""


class ConfigurationLoadError(ConfigurationError):
    def __init__(self, config_file: str, original_error: Exception):
        self.config_file = config_file
        self.original_error = original_error
        message = f"Error loading configuration from {config_file}: {original_error!s}"
        super().__init__(message)


class InvalidPricingConfigError(ConfigurationError):
    def __init__(self):
        message = "Either token_prices or character_prices must be set"
        super().__init__(message)


class ProviderNotFoundError(Exception):
    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        message = f"Provider '{provider_id}' not found"
        super().__init__(message)
