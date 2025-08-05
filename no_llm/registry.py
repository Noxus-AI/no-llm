from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from no_llm.models.registry import ModelRegistry
from no_llm.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from no_llm.models.config import ModelConfiguration
    from no_llm.providers.config import ProviderConfiguration


class Registry:
    def __init__(self, config_dir: str | Path | None = None):
        logger.debug("Initializing main Registry")
        self.models = ModelRegistry(config_dir)
        self.providers = ProviderRegistry(config_dir)

    def get_compatible_providers(
        self, model_id: str, *, only_valid: bool = True, only_active: bool = True
    ) -> Iterator[ProviderConfiguration]:
        """Get all providers compatible with a specific model

        Args:
            model_id: The model ID to find compatible providers for
            only_valid: If True, only return providers with valid environment setup
            only_active: If True, only return providers that are active
        """
        logger.debug(f"Finding compatible providers for model: {model_id}")

        model = self.models.get(model_id)

        # Get provider types that this model supports
        compatible_provider_types = {provider().type for provider in model._compatible_providers}  # noqa: SLF001
        logger.debug(f"Model {model_id} supports provider types: {compatible_provider_types}")

        # Find all providers of compatible types
        for provider_type in compatible_provider_types:
            yield from self.providers.list_by_type(provider_type, only_valid=only_valid, only_active=only_active)

    def get_models_for_provider(
        self, provider_id: str, *, only_valid: bool = True, only_active: bool = True
    ) -> Iterator[ModelConfiguration]:
        """Get all models that can use a specific provider

        Args:
            provider_id: The provider ID to find compatible models for
            only_valid: If True, only return models with valid configuration
            only_active: If True, only return models that are active
        """
        logger.debug(f"Finding models compatible with provider: {provider_id}")

        provider = self.providers.get(provider_id)
        logger.debug(f"Provider {provider_id} is of type: {provider.type}")

        # Find all models that support this provider type
        for model in self.models.list(only_valid=only_valid, only_active=only_active):
            if type(provider) in model._compatible_providers:  # noqa: SLF001
                logger.debug(f"Model {model.identity.id} is compatible with provider {provider_id}")
                yield model

    def reload(self) -> None:
        """Reload all registry configurations"""
        logger.debug("Reloading all registries")
        self.models.reload()
        self.providers.reload()
