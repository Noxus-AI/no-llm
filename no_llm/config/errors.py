from __future__ import annotations

from typing import Any


class ParameterError(Exception):
    def __init__(self, param_name: str, message: str, description: str | None = None):
        self.param_name = param_name
        self.description = description

        full_message = message
        if description:
            full_message += f"\nDescription: {description}"

        super().__init__(full_message)


class FixedParameterError(ParameterError):
    def __init__(self, param_name: str, current_value: Any, attempted_value: Any, description: str | None = None):
        self.current_value = current_value
        self.attempted_value = attempted_value

        message = (
            f"Cannot modify fixed parameter '{param_name}'\n"
            f"Fixed value: {current_value}\n"
            f"Attempted value: {attempted_value}"
        )
        super().__init__(param_name, message, description)


class UnsupportedParameterError(ParameterError):
    def __init__(self, param_name: str, required_capability: str | None = None, description: str | None = None):
        self.required_capability = required_capability

        message = f"Parameter '{param_name}' is not supported by this model"
        if required_capability:
            message += f"\nRequired capability: {required_capability}"
            message += "\nModel does not support the required capability"

        super().__init__(param_name, message, description)


class InvalidRangeError(ParameterError):
    def __init__(self, param_name: str, value: Any, reason: str, valid_range: tuple[float | int, float | int]):
        self.value = value
        self.reason = reason
        self.valid_range = valid_range
        self._param_name = param_name

        self._update_message()

    @property
    def param_name(self) -> str:
        return self._param_name

    @param_name.setter
    def param_name(self, value: str):
        self._param_name = value
        self._update_message()

    def _update_message(self):
        message = [f"Invalid value for parameter '{self._param_name}'"]
        message.append(f"Current value: {self.value}")
        if self.valid_range:
            message.append(f"Valid range: {self.valid_range}")
        message.append(f"Error: {self.reason}")

        Exception.__init__(self, "\n".join(message))


class InvalidEnumError(ParameterError):
    def __init__(self, param_name: str, value: Any, reason: str, valid_values: list[Any]):
        self.value = value
        self.reason = reason
        self.valid_values = valid_values
        self._param_name = param_name
        self._update_message()

    @property
    def param_name(self) -> str:
        return self._param_name

    @param_name.setter
    def param_name(self, value: str):
        self._param_name = value
        self._update_message()

    def _update_message(self):
        message = [f"Invalid value for parameter '{self._param_name}'"]
        message.append(f"Current value: {self.value}")
        if self.valid_values:
            message.append(f"Valid values: {self.valid_values}")
        message.append(f"Error: {self.reason}")

        Exception.__init__(self, "\n".join(message))


class MissingCapabilitiesError(Exception):
    def __init__(self, model_name: str, required_capabilities: list[str], available_capabilities: list[str]):
        self.model_name = model_name
        self.required_capabilities = required_capabilities
        self.available_capabilities = available_capabilities
