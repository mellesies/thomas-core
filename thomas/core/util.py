# -*- coding: utf-8 -*-
"""utilities"""

# def sep(default='-', nchar=80):
#     print(default * nchar)

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
flatten = lambda l: [item for sublist in l for item in sublist]
