import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

"""
Load data from the__versions__.py module. Change version, etc in
that module, and it will be automatically populated here. This allows us to
access the module version, etc from inside python with

Examples:

    >>> from wellcomeml.common import about
    >>> about['__version__']
    2019.10.0

"""

about = {}  # type: dict
version_path = os.path.join(here, 'wellcomeml', '__version__.py')
with open(version_path, 'r') as f:
    exec(f.read(), about)

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=about['__name__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=about['__url__'],
    license=['__license__'],
    packages=setuptools.find_packages(include=["wellcomeml*"]),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy<1.19.0,>=1.16.0',
        'pandas',
        'boto3',
        'scikit-learn',
        'scipy==1.4.1',
        'spacy==2.2.1',
        'umap-learn',
        'nervaluate',
        'twine',
        'gensim',
        'hdbscan',
        'cython',
        'flake8',
        'black'
    ],
    extras_require={
        'deep-learning': [
            'tensorflow',
            'torch',
            'transformers<=2.0.0',
            'spacy-transformers==0.5.1'
        ]
    },
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
)
