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
        'numpy==1.20.0',
        'pandas==1.2.1',
        'boto3==1.16.63',
        'scikit-learn==0.24.1',
        'scipy==1.4.1',
        'spacy==2.3.5',
        'umap-learn==0.5.0',
        'nervaluate==0.1.8',
        'twine==3.3.0',
        'gensim==3.8.3',
        'hdbscan==0.8.26',
        'cython',
        'flake8==3.8.4',
        'black==20.8b1',
        'tqdm==4.56.0'
    ],
    extras_require={
        'deep-learning': [
            'tensorflow==2.4.0',
            'torch==1.7.1',
            'transformers<2.9.0',
            'spacy-transformers==0.6.1',
            'dataclasses==0.6'  # spacy transformers needs this pinned
        ]
    },
    tests_require=[
        'pytest==6.2.2',
        'pytest-cov==2.11.1'
    ]
)
