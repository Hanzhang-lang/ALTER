# -*- coding: utf-8 -*-
from __future__ import absolute_import

"""
Various utilities
"""

class Const(object):
    """
    Wrapper for a set of constants
    """

    # def __new__(cls, name=None, docstring=None, **const):
    #     attr = {}
    #     if docstring:
    #         attr['__doc__'] = docstring
    #     Class = super().__new__(cls, name=name, docstring=docstring, **const)
    #     return Class, attr

    def __init__(self, name=None, docstring=None, **const):
        super().__init__()
        self.name = name
        self.docstring=docstring
        self.__dict__.update(const)

    def __contains__(self, value):
        """
        Test if `value` is a valid constant in this set
        """
        return value in self.__dict__