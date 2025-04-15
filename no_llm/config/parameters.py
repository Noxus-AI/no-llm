from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from loguru import logger
from pydantic import BaseModel, Field, model_serializer, model_validator

from no_llm.config.enums import ModelCapability
from no_llm.config.errors import FixedParameterError, InvalidEnumError, InvalidRangeError, UnsupportedParameterError
from no_llm.settings import ValidationMode
from no_llm.settings import settings as no_llm_settings

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
                param_name="value", value=value, reason="Value not in allowed values", valid_values=self.allowed_values
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
        return cls(variant=ParameterVariant.VARIABLE, value=value, required_capability=required_capability)

    def check_capability(self, capabilities: set[ModelCapability]) -> ParameterValue[V]:
        """Check if this parameter is supported given the capabilities.
        Returns a new ParameterValue with variant=UNSUPPORTED if not supported.
        """
        if self.required_capability and self.required_capability not in capabilities:
            return ParameterValue(
                variant=ParameterVariant.UNSUPPORTED, value=None, required_capability=self.required_capability
            )
        return self

    def validate_new_value(self, new_value: V, field_name: str) -> None:
        """Validate a new value against this parameter's constraints"""
        if self.is_fixed():
            raise FixedParameterError(
                param_name=field_name, current_value=self.value, attempted_value=new_value, description=None
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
            result["range"] = [self.validation_rule.min_value, self.validation_rule.max_value]
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
            variant=ParameterVariant.VARIABLE, value=NOT_GIVEN, required_capability=ModelCapability.REASONING
        ),
        description="Whether to include reasoning steps",
    )
    reasoning_effort: ParameterValue[Literal["low", "medium", "high"] | NotGiven] = Field(
        default_factory=lambda: ParameterValue[Literal["low", "medium", "high"] | NotGiven](
            variant=ParameterVariant.VARIABLE, value=NOT_GIVEN, required_capability=ModelCapability.REASONING
        ),
        description="Reasoning level",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_yaml(cls, data: dict[str, Any]) -> dict[str, Any]:
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
                            data[field_name] = {"variant": ParameterVariant.UNSUPPORTED, "value": None}
                            continue

                        if not isinstance(value, dict):
                            data[field_name] = {"variant": ParameterVariant.FIXED, "value": value}
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

        return data

    def get_parameters(self) -> dict[str, Any]:
        values = {}
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ParameterValue) and not value.is_unsupported():
                gotten_value = value.get()
                if gotten_value is not None:
                    values[field_name] = gotten_value
        return values

    def validate_parameters(self, drop_unsupported: bool = True, **kwargs) -> dict[str, Any]:
        capabilities = kwargs.pop("capabilities", set())
        settings = {}

        for field_name, field in self.model_fields.items():
            value = getattr(self, field_name)
            description = field.description

            if isinstance(value, ParameterValue):
                value = value.check_capability(capabilities)

                if value.is_unsupported():
                    if field_name in kwargs:
                        if drop_unsupported:
                            logger.warning(f"Dropping unsupported parameter: {field_name}")
                            kwargs.pop(field_name, None)
                        else:
                            raise UnsupportedParameterError(
                                param_name=field_name,
                                required_capability=str(value.required_capability),
                                description=description,
                            )
                    continue

                if field_name in kwargs:
                    try:
                        if kwargs[field_name] != NOT_GIVEN:
                            value.validate_new_value(kwargs[field_name], field_name)
                        settings[field_name] = kwargs[field_name]
                    except (FixedParameterError, InvalidRangeError, InvalidEnumError) as e:
                        if isinstance(e, InvalidEnumError | InvalidRangeError):
                            e.param_name = field_name

                        if no_llm_settings.validation_mode == ValidationMode.ERROR:
                            raise

                        if isinstance(e, InvalidRangeError):
                            if no_llm_settings.validation_mode == ValidationMode.CLAMP:
                                logger.warning(
                                    f"Clamping invalid parameter value for {field_name}: {kwargs[field_name]}"
                                )
                                settings[field_name] = (
                                    e.valid_range[0]
                                    if kwargs[field_name] < e.valid_range[0]
                                    else e.valid_range[1]
                                    if kwargs[field_name] > e.valid_range[1]
                                    else kwargs[field_name]
                                )
                        else:
                            logger.warning(f"Invalid parameter value for {field_name}: {e}")
                else:
                    # Convert None values to NOT_GIVEN
                    gotten_value = value.get()
                    settings[field_name] = NOT_GIVEN if gotten_value is None else gotten_value

                kwargs.pop(field_name, None)
        settings.update(kwargs)

        return settings

    def set_parameters(self, capabilities: set[ModelCapability] | None = None, **kwargs) -> None:
        validated = self.validate_parameters(capabilities=capabilities or set(), drop_unsupported=True, **kwargs)

        for field_name, new_value in validated.items():
            if hasattr(self, field_name):
                param_value = getattr(self, field_name)
                if isinstance(param_value, ParameterValue):
                    # Create new ParameterValue with updated value but same validation rules
                    setattr(
                        self,
                        field_name,
                        ParameterValue(
                            variant=ParameterVariant.VARIABLE,
                            value=new_value,
                            validation_rule=param_value.validation_rule,
                            required_capability=param_value.required_capability,
                        ),
                    )

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        result = {}
        for field_name, field in self.model_fields.items():
            value = getattr(self, field_name)
            if isinstance(value, ParameterValue):
                default_factory = getattr(field, "default_factory", None)
                default_value = default_factory() if default_factory else None

                # Include if unsupported or has non-default, non-null value
                if value.is_unsupported() or (
                    value.value is not None and (default_value is None or value.value != default_value.value)
                ):
                    result[field_name] = value
        return result

    def get_model_parameters(self) -> ModelParameters:
        return ModelParameters(**self.get_parameters())


class ModelParameters(BaseModel):
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
    model_override: dict[str, ModelParameters] | NotGiven = Field(
        default=NOT_GIVEN,
        description="Model override parameters",
    )

    def __and__(self, other: ModelParameters) -> ModelParameters:
        """Merge two ModelParameters objects with right-hand overrides"""
        return ModelParameters(
            **{**other.dump_parameters(with_defaults=False), **self.dump_parameters(with_defaults=False)}
        )

    def get_parameters(self) -> dict[str, Any]:
        """Get all parameter values"""
        return self.model_dump(exclude_defaults=True)

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
