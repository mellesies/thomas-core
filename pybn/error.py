# -*- coding: utf-8 -*-
"""Errors and Exceptions"""

class NotInScopeError(Exception):
    def __init__(self, variable, scope):
        msg = f"Variable {variable} not in scope {scope}"
        super().__init__(msg)
