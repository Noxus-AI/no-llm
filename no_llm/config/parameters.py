from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from no_llm.config.enums import ModelCapability
from no_llm.config.errors import (
    FixedParameterError,
    InvalidEnumError,
    InvalidRangeError,
    UnsupportedParameterError,
)
from no_llm.settings import ValidationMode
from no_llm.settings import settings as no_llm_settings

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings

V = TypeVar("V")
NotGiven = Literal["NOT_GIVEN"]
NOT_GIVEN: NotGiven = "NOT_GIVEN"


class ParameterVariant(str, Enum):
    FIXED = "fixed"
    VARIABLE = "variable"
    UNSUPPORTED = "unsupported"


class ValidationRule(BaseModel):
    def validate_value(self, value: Any) -> None:
        pass


class RangeValidation(ValidationRule):
    min_value: float | int
    max_value: float | int

    def validate_value(self, value: Any) -> None:
        if value == NOT_GIVEN:
            return

        if not (self.min_value <= value <= self.max_value):
            raise InvalidRangeError(
                param_name="value",
                value=value,
                reason=f"Value {value} outside range [{self.min_value}, {self.max_value}]",
                valid_range=(self.min_value, self.max_value),
            )


class EnumValidation(ValidationRule):
    allowed_values: list[Any]

    def validate_value(self, value: Any) -> None:
        if value == NOT_GIVEN:
            return

        if value not in self.allowed_values:
            raise InvalidEnumError(
                param_name="value",
                value=value,
                reason="Value not in allowed values",
                valid_values=self.allowed_values,
            )


class ParameterValue(BaseModel, Generic[V]):
    """A parameter value that can be fixed, variable, or unsupported.

    YAML formats:
        0.7                 # Shorthand for fixed value
        unsupported        # Shorthand for unsupported parameter

        # Variable with range validation
        value: 0.7
        range: [0.0, 2.0]  # min, max inclusive

        # Variable with enum validation
        value: "medium"
        values: ["low", "medium", "high"]

        # Explicit formats still supported
        fixed: 0.7
        variable: 0.7
        unsupported: true
    """

    variant: ParameterVariant
    value: V | None = None
    validation_rule: RangeValidation | EnumValidation | None = None
    required_capability: ModelCapability | None = None

    @model_validator(mode="after")
    def validate_model(self) -> ParameterValue[V]:
        if self.validation_rule is not None and self.value is not None and self.value != NOT_GIVEN:
            self.validation_rule.validate_value(self.value)
        return self

    def get(self) -> V | None | Literal["UNSUPPORTED"]:
        """Get the parameter value."""
        if self.variant == ParameterVariant.UNSUPPORTED:
            return "UNSUPPORTED"
        return self.value

    def is_fixed(self) -> bool:
        return self.variant == ParameterVariant.FIXED

    def is_variable(self) -> bool:
        return self.variant == ParameterVariant.VARIABLE

    def is_unsupported(self) -> bool:
        return self.variant == ParameterVariant.UNSUPPORTED

    @classmethod
    def create_variable(cls, value: V, required_capability: ModelCapability | None = None) -> ParameterValue[V]:
        return cls(
            variant=ParameterVariant.VARIABLE,
            value=value,
            required_capability=required_capability,
        )

    def check_capability(self, capabilities: set[ModelCapability]) -> ParameterValue[V]:
        """Check if this parameter is supported given the capabilities.
        Returns a new ParameterValue with variant=UNSUPPORTED if not supported.
        """
        if self.required_capability and self.required_capability not in capabilities:
            return ParameterValue(
                variant=ParameterVariant.UNSUPPORTED,
                value=None,
                required_capability=self.required_capability,
            )
        return self

    def validate_new_value(self, new_value: V, field_name: str) -> None:
        """Validate a new value against this parameter's constraints"""
        if self.is_fixed() and new_value != self.value:
            raise FixedParameterError(
                param_name=field_name,
                current_value=self.value,
                attempted_value=new_value,
                description=None,
            )
        if new_value != NOT_GIVEN and self.validation_rule is not None:
            self.validation_rule.validate_value(new_value)

    @model_serializer
    def serialize_model(self) -> V | dict[str, Any] | str:
        """Custom serialization for ParameterValue"""
        if self.is_unsupported():
            return "unsupported"

        if self.is_fixed():
            return self.value  # type: ignore

        result: dict[str, Any] = {"value": self.value}

        if isinstance(self.validation_rule, RangeValidation):
            result["range"] = [
                self.validation_rule.min_value,
                self.validation_rule.max_value,
            ]
        elif isinstance(self.validation_rule, EnumValidation):
            result["values"] = self.validation_rule.allowed_values

        return result


class ConfigurableModelParameters(BaseModel):
    """Complete set of model parameters"""

    temperature: ParameterValue[float | NotGiven] = Field(
        default_factory=lambda: ParameterValue[float | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
        ),
        description="Controls randomness in generation",
    )
    top_p: ParameterValue[float | NotGiven] = Field(
        default_factory=lambda: ParameterValue[float | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
        ),
        description="Nucleus sampling threshold",
    )
    top_k: ParameterValue[int | NotGiven] = Field(
        default_factory=lambda: ParameterValue[int | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0, max_value=float("inf")),
        ),
        description="Top-k sampling threshold",
    )

    frequency_penalty: ParameterValue[float | NotGiven] = Field(
        default_factory=lambda: ParameterValue[float | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=-2.0, max_value=2.0),
        ),
        description="Penalty for token frequency",
    )
    presence_penalty: ParameterValue[float | NotGiven] = Field(
        default_factory=lambda: ParameterValue[float | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=-2.0, max_value=2.0),
        ),
        description="Penalty for token presence",
    )
    logit_bias: ParameterValue[dict[str, float] | NotGiven] = Field(
        default_factory=lambda: ParameterValue[dict[str, float] | NotGiven](
            variant=ParameterVariant.VARIABLE, value=NOT_GIVEN
        ),
        description="Token biasing dictionary",
    )
    max_tokens: ParameterValue[int | NotGiven] = Field(
        default_factory=lambda: ParameterValue[int | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=1, max_value=float("inf")),
        ),
        description="Maximum number of tokens to generate",
    )
    stop: ParameterValue[list[str] | NotGiven] = Field(
        default_factory=lambda: ParameterValue[list[str] | NotGiven](
            variant=ParameterVariant.VARIABLE, value=NOT_GIVEN
        ),
        description="Stop sequences",
    )
    logprobs: ParameterValue[int | NotGiven] = Field(
        default_factory=lambda: ParameterValue[int | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0, max_value=float("inf")),
        ),
        description="Number of logprobs to return",
    )
    top_logprobs: ParameterValue[int | NotGiven] = Field(
        default_factory=lambda: ParameterValue[int | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0, max_value=float("inf")),
        ),
        description="Number of most likely tokens to return",
    )
    seed: ParameterValue[int | NotGiven] = Field(
        default_factory=lambda: ParameterValue[int | NotGiven](variant=ParameterVariant.VARIABLE, value=NOT_GIVEN),
        description="Random seed for reproducibility",
    )
    timeout: ParameterValue[float | NotGiven] = Field(
        default_factory=lambda: ParameterValue[float | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            validation_rule=RangeValidation(min_value=0.0, max_value=float("inf")),
        ),
        description="Request timeout in seconds",
    )

    include_reasoning: ParameterValue[bool | NotGiven] = Field(
        default_factory=lambda: ParameterValue[bool | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            required_capability=ModelCapability.REASONING,
        ),
        description="Whether to include reasoning steps",
    )
    reasoning_effort: ParameterValue[Literal["low", "medium", "high"] | NotGiven] = Field(
        default_factory=lambda: ParameterValue[Literal["low", "medium", "high"] | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            required_capability=ModelCapability.REASONING,
        ),
        description="Reasoning level",
    )
    parallel_tool_calls: ParameterValue[bool | NotGiven] = Field(
        default_factory=lambda: ParameterValue[bool | NotGiven](
            variant=ParameterVariant.VARIABLE,
            value=NOT_GIVEN,
            required_capability=ModelCapability.PARALLEL_FUNCTION_CALLING,
        ),
        description="Whether to allow parallel tool calling",
    )

    # @model_validator(mode="before")
    @classmethod
    def from_config(cls, data: dict[str, Any]) -> ConfigurableModelParameters:
        """Extract capabilities from field defaults and handle YAML parsing."""
        if not isinstance(data, dict):
            return data

        for field_name, field in cls.model_fields.items():
            # Get capability and validation rule from field's default factory if it exists
            default_factory = getattr(field, "default_factory", None)
            if default_factory is not None and callable(default_factory):
                default = default_factory()
                if isinstance(default, ParameterValue):  # noqa: SIM102
                    # If we have a value in the data, parse it and add capabilities
                    if field_name in data:
                        value = data[field_name]

                        # Handle shorthand formats
                        if value == "unsupported":
                            data[field_name] = {
                                "variant": ParameterVariant.UNSUPPORTED,
                                "value": None,
                            }
                            continue

                        if not isinstance(value, dict):
                            data[field_name] = {
                                "variant": ParameterVariant.FIXED,
                                "value": value,
                            }
                            continue

                        result = {}

                        # Handle variant
                        if "variant" in value:
                            result["variant"] = value["variant"]
                        elif "fixed" in value:
                            result["variant"] = ParameterVariant.FIXED
                            result["value"] = value["fixed"]
                        else:
                            result["variant"] = ParameterVariant.VARIABLE

                        # Handle value
                        if "value" in value:
                            result["value"] = value["value"]

                        # Handle validation rule
                        if "range" in value:
                            min_val, max_val = value["range"]
                            result["validation_rule"] = RangeValidation(min_value=min_val, max_value=max_val)
                        elif default.validation_rule:
                            result["validation_rule"] = default.validation_rule

                        # Handle capability
                        if "required_capability" in value:
                            result["required_capability"] = value["required_capability"]
                        elif default.required_capability:
                            result["required_capability"] = default.required_capability

                        data[field_name] = result

        return cls(**data)

    def validate_parameter(
        self,
        field_name: str,
        value: Any,
        capabilities: set[ModelCapability] | None = None,
    ) -> Any:
        """Validate a single parameter value, raising errors if invalid."""
        if not hasattr(self, field_name):
            msg = f"Unknown parameter: {field_name}"
            raise ValueError(msg)

        current_value = getattr(self, field_name)
        if not isinstance(current_value, ParameterValue):
            return value

        # Check capabilities
        current_value = current_value.check_capability(capabilities or set())
        if current_value.is_unsupported():
            raise UnsupportedParameterError(
                param_name=field_name,
                required_capability=str(current_value.required_capability),
                description=self.model_fields[field_name].description,
            )

        # Validate the value
        current_value.validate_new_value(value, field_name)
        return value

    def _handle_validation_error(
        self,
        error: Exception,
        field_name: str,
        value: Any,
        param_value: ParameterValue[Any],
    ) -> ParameterValue[Any] | None:
        """Handle validation errors according to validation mode settings.
        Returns new ParameterValue if value should be updated, None otherwise."""
        if isinstance(error, FixedParameterError):
            if no_llm_settings.validation_mode == ValidationMode.ERROR:
                raise error
            logger.warning(f"Invalid parameter value for {field_name}: {error}")
            return None

        if isinstance(error, InvalidRangeError):
            if no_llm_settings.validation_mode == ValidationMode.ERROR:
                raise error
            if no_llm_settings.validation_mode == ValidationMode.WARN:
                logger.warning(f"Invalid parameter value for {field_name}: {error}")
                return None
            if no_llm_settings.validation_mode == ValidationMode.CLAMP:
                logger.warning(f"Clamping invalid parameter value for {field_name}: {error}")
                clamped_value = error.valid_range[0] if value < error.valid_range[0] else error.valid_range[1]
                return ParameterValue(
                    variant=param_value.variant,
                    value=clamped_value,
                    validation_rule=param_value.validation_rule,
                    required_capability=param_value.required_capability,
                )
            return None

        if isinstance(error, InvalidEnumError | UnsupportedParameterError):
            if no_llm_settings.validation_mode == ValidationMode.ERROR:
                raise error
            logger.warning(f"Invalid parameter value for {field_name}: {error}")
            return None

        raise error  # Re-raise unexpected errors

    def _validate_and_update_parameter(
        self,
        field_name: str,
        value: Any,
        capabilities: set[ModelCapability] | None = None,
    ) -> None:
        """Validate and update a parameter value, handling any validation errors."""
        current_value = getattr(self, field_name)
        try:
            validated_value = self.validate_parameter(field_name, value, capabilities)
            if isinstance(current_value, ParameterValue):
                super().__setattr__(
                    field_name,
                    ParameterValue(
                        variant=current_value.variant,
                        value=validated_value,
                        validation_rule=current_value.validation_rule,
                        required_capability=current_value.required_capability,
                    ),
                )
        except (
            FixedParameterError,
            InvalidRangeError,
            InvalidEnumError,
            UnsupportedParameterError,
        ) as e:
            new_value = self._handle_validation_error(e, field_name, value, current_value)
            if new_value is not None:
                super().__setattr__(field_name, new_value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to validate parameters when set directly."""
        if name in self.model_fields:
            if isinstance(value, ParameterValue):
                super().__setattr__(name, value)
                return
            self._validate_and_update_parameter(name, value)
            return
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def validate_parameters(self) -> ConfigurableModelParameters:
        """Validate all parameters during model initialization."""
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ParameterValue) and value.value != NOT_GIVEN:
                self._validate_and_update_parameter(field_name, value.value)
        return self

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ParameterValue):
                # Skip unsupported parameters
                if value.is_unsupported():
                    continue

                # Skip NOT_GIVEN values
                if value.value == NOT_GIVEN:
                    continue

                # Include the current value for all supported parameters
                result[field_name] = value.value

        return result


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature: float | NotGiven = Field(
        default=NOT_GIVEN,
        description="Controls randomness in generation",
    )
    top_p: float | NotGiven = Field(
        default=NOT_GIVEN,
        description="Nucleus sampling threshold",
    )
    top_k: int | NotGiven = Field(
        default=NOT_GIVEN,
        description="Top-k sampling threshold",
    )
    frequency_penalty: float | NotGiven = Field(
        default=NOT_GIVEN,
        description="Penalty for token frequency",
    )
    presence_penalty: float | NotGiven = Field(
        default=NOT_GIVEN,
        description="Penalty for token presence",
    )
    logit_bias: dict[str, float] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Token biasing dictionary",
    )
    max_tokens: int | NotGiven = Field(
        default=NOT_GIVEN,
        description="Maximum number of tokens to generate",
    )
    stop: list[str] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Stop sequences",
    )
    logprobs: int | NotGiven = Field(
        default=NOT_GIVEN,
        description="Number of logprobs to return",
    )
    top_logprobs: int | NotGiven = Field(
        default=NOT_GIVEN,
        description="Number of most likely tokens to return",
    )
    seed: int | NotGiven = Field(
        default=NOT_GIVEN,
        description="Random seed for reproducibility",
    )
    timeout: float | NotGiven = Field(
        default=NOT_GIVEN,
        description="Request timeout in seconds",
    )
    include_reasoning: bool | NotGiven = Field(
        default=NOT_GIVEN,
        description="Whether to include reasoning steps",
    )
    reasoning_effort: Literal["low", "medium", "high"] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Reasoning level",
    )
    parallel_tool_calls: bool | NotGiven = Field(
        default=NOT_GIVEN, description="Whether to allow parallel function calling"
    )
    model_override: dict[str, ModelParameters] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Model override parameters",
    )

    def __and__(self, other: ModelParameters) -> ModelParameters:
        """Merge two ModelParameters objects with right-hand overrides"""
        return ModelParameters(
            **{
                **other.dump_parameters(with_defaults=False),
                **self.dump_parameters(with_defaults=False),
            }
        )

    def dump_parameters(self, with_defaults: bool = False, model_override: str | None = None) -> dict[str, Any]:
        """Get all parameter values"""
        params = self.model_dump(exclude_defaults=not with_defaults)
        if (
            model_override is not None
            and self.model_override != NOT_GIVEN
            and isinstance(self.model_override, dict)
            and model_override in self.model_override
        ):
            override_params = self.model_override[model_override].dump_parameters(with_defaults=not with_defaults)
            params.update(override_params)
        return params

    @classmethod
    def from_pydantic(cls, model_settings: ModelSettings) -> ModelParameters:
        if "extra_body" in model_settings:
            model_settings.pop("extra_body")
        if "extra_headers" in model_settings:
            model_settings.pop("extra_headers")
        if "stop_sequences" in model_settings:
            model_settings.pop("stop_sequences")
        extra = {**model_settings}
        # if "openai_reasoning_effort" in model_settings:
        #     extra["reasoning_effort"] = model_settings.pop("openai_reasoning_effort")
        return ModelParameters(**extra)
