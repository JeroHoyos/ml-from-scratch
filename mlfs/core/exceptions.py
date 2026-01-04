"""Custom exceptions for the ML framework."""


class NotFittedError(Exception):
    """Exception raised when model is used before fitting."""
    pass


class InvalidInputError(Exception):
    """Exception raised for invalid input."""
    pass


class InvalidShapeError(InvalidInputError):
    """Exception raised for invalid array shapes."""
    pass


class InvalidTypeError(InvalidInputError):
    """Exception raised for invalid data types."""
    pass
