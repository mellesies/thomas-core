# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages

# To use a consistent encoding
from codecs import open
import os
from os import path

here = path.abspath(path.dirname(__file__))
PKG_NAME = 'thomas-core'
PKG_DESC = 'Thomas, a library for working with Bayesian Networks.'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    PKG_DESCRIPTION = f.read()


# Read the API version from disk. This file should be located in the package
# folder, since it's also used to set the pkg.__version__ variable.
version_path = os.path.join(here, 'thomas', 'core', '_version.py')
version_ns = {
    '__file__': version_path
}
with open(version_path) as f:
    exec(f.read(), {}, version_ns)


# Setup the package
setup(
    name=PKG_NAME,
    version=version_ns['__version__'],
    description=PKG_DESC,
    long_description=PKG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/mellesies/thomas-core',
    author='Melle Sieswerda',
    author_email='m.sieswerda@iknl.nl',
    packages=find_namespace_packages(include=['thomas.*']),
    package_data={
        "thomas.core": [
            "__build__",
            "data/dataset_17_2.csv",
            "data/dataset_17_2_with_NAs.csv",
            "data/dataset_17_3.csv",
            "data/*.lark",
            "data/*.json",
            "data/*.oobn",
            "data/*.net",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>= 3.6',
    install_requires=[
        'lark-parser',
        'matplotlib>=3.1',
        'networkx>=2.4',
        'numpy>=1.18',
        'pandas>=1',
        'termcolor',
    ],
    # This is incompatible with publication on PyPI ...
    # extras_require={
    #     "jupyter-dev": [
    #         "thomas-jupyter-widget @ git+https://github.com/mellesies/thomas-jupyter-widget@jupyter3",
    #     ],
    #     "client": [
    #         "thomas-client @ git+https://github.com/mellesies/thomas-client",
    #     ],
    # }
)
