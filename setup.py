# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
PKG_NAME = 'pybn'
PKG_DESC = 'Very simple (almost naive ;) bayesian network implementation'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the API version from disk
with open(path.join(here, PKG_NAME, 'VERSION')) as fp:
    __version__ = fp.read()



setup(
    name=PKG_NAME,
    version=__version__,
    description=PKG_DESC,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mellesies/py-bn',
    author='Melle Sieswerda',
    author_email='m.sieswerda@iknl.nl',
    license='Apache 2.0',
    packages=['pybn'],
    python_requires='>= 3.6',
    install_requires=[
        'lark-parser',
        'networkx',
        'numpy',
        'pandas',
        'termcolor',
    ],
)