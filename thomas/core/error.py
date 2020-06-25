# -*- coding: utf-8 -*-
"""Errors and Exceptions"""

class NotInScopeError(Exception):
    def __init__(self, variable, scope):
        msg = f"Variable {variable} not in scope {scope}"
        super().__init__(msg)


class InvalidStateError(Exception):
    def __init__(self, variable, state, factor):
        name = factor.display_name
        msg = f"State '{state}' is invalid for variable '{variable}'"
        super().__init__(msg)

class NoStatesOrIndexProvidedError(Exception):
    """Raised when Factor() is called but no index was provided."""

    def __init__(self):
        msg = f"Insufficient information to create Factor! "
        msg += f"Either 'states' or 'idx' should be provided!"
        super().__init__(msg)


class InvalidCPTError(Exception):
    def __init__(self, msg):
        super().__init__(msg)