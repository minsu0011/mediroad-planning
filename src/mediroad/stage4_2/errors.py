from __future__ import annotations

from typing import Any


class Stage42Error(RuntimeError):
    """Base error for Stage 4.2."""


class ContractError(Stage42Error):
    """Raised when a frozen input or schema contract is violated."""


class InputNotReadyError(Stage42Error):
    """Raised when required operational evidence is not yet available."""


class SolverCertificationError(Stage42Error):
    """Raised when a solution cannot be certified under the frozen gap contract."""

    def __init__(self, message: str, *, telemetry: list[Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = list(telemetry or [])
