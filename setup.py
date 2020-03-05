# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
PKG_NAME = 'thomas-core'
PKG_DESC = 'Simple (almost naive ;) bayesian network implementation'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    PKG_DESCRIPTION = f.read()


# Read the API version from disk. This file should be located in the package
# folder, since it's also used to set the pkg.__version__ variable.
with open(path.join(here, 'thomas', 'core', 'VERSION')) as fp:
    PKG_VERSION = fp.read()


# Setup the package
setup(
    name=PKG_NAME,
    version=PKG_VERSION,
    description=PKG_DESC,
    long_description=PKG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/mellesies/thomas-core',
    author='Melle Sieswerda',
    author_email='m.sieswerda@iknl.nl',
    license='Apache 2.0',
    packages=find_namespace_packages(include=['thomas.*']),
    package_data={
        "thomas.core": [
            "VERSION",
            "data/*.lark",
            "data/*.json",
            "data/*.oobn",
        ],
    },
    python_requires='>= 3.6',
    install_requires=[
        'lark-parser',
        'networkx',
        'numpy',
        'pandas',
        'termcolor',
        'matplotlib',
    ],
)