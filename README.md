Lentil - Latent Skill Embedding
===============================

A package for training, evaluation, and visualization of the Latent Skill
Embedding model with interaction log data. Read more about the model at
http://siddharth.io/research.

Usage
-----

You can install the package in your environment with

```
python setup.py install
```

If you wish to run the tests, simply make sure you have
[tox](https://tox.readthedocs.org/en/latest/) installed and then simply run

```
tox
```

Once installed in your environment, command-line interfaces for training and
evaluation are available through `train.py` and `evaluate.py`. IPython
notebooks used to conduct experiments are available in the source directory,
and provide example invocations of most functions and classes.

QUESTION: There are no notebooks in this repo

Documentation
-------------

Build the documentation with

```
tox -e docs
```

Once run, open doc/_build/html/index.html for Sphinx documentation on the
various modules in the package.

Questions and comments
----------------------

Please contact the author if you have questions or find bugs.
