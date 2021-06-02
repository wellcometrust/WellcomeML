.. WellcomeML documentation master file, created by
   sphinx-quickstart on Mon Jun 22 09:56:00 2020.

WellcomeML's documentation!
======================================

Current release: |release|

This package contains common utility functions for usual tasks at Wellcome Data
Labs, in particular functionalities for processing, embedding and classifying text data.
This includes

* An intuitive sklearn-like API wrapping text vectorizers, such as Doc2vec, Bert, Scibert
* Common API for off-the-shelf classifiers to allow quick iteration (e.g. Frequency Vectorizer, Bert, Scibert, basic CNN, BiLSTM)
* Utils to download and convert academic text datasets for benchmark

Check :ref:`examples` for some examples and :ref:`clustering` for clustering-specific documentation.

Quickstart
-------------------------------------

In order to install the latest release, with all the deep learning functionalities::

    pip install wellcomeml[deep-learning]

For a quicker installation that only includes certain frequency vectorisers, the io operations
and the spacy-to-prodigy conversions::

   pip install wellcomeml


Development
-------------------------------------

For installing the latest main branch::

    pip install git+https://github.com/wellcometrust/WellcomeML.git[deep-learning]

If you want to contribute, please refer to the issues and documentation in the main `github repository <https://github.com/wellcometrust/WellcomeML>`_

Contact
--------
To contact us, you can open an issue in the main github repository or e mail `Data Labs <DataLabs@wellcome.ac.uk>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:

   self
   
.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   Examples <examples>
   Clustering text with WellcomeML <clustering>
   Core library documentation <wellcomeml>
