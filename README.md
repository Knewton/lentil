Lentil - Latent Skill Embedding
===============================

A package for training, evaluation, and visualization of the Latent Skill Embedding model. Read 
more about the model at http://siddharth.io/lentil.

Usage
-----

You can install the package's dependencies with 

```
pip install -r requirements.txt
```

You can install the package in your environment with

```
python setup.py install
```

If you wish to run the tests, make sure you have
[tox](https://tox.readthedocs.org/en/latest/) installed and then run

```
tox
```

Once installed in your environment, command-line interfaces for training and
evaluation are available through `lse_train` and `lse_eval`. The appropriate format for input
interaction log data is given in the documentation for `lentil.datatools.InteractionHistory`.
IPython notebooks used to conduct experiments are available in the `nb` directory, and provide 
example invocations of most functions and classes. It is recommended that you read the notebooks 
in the following order: `toy_examples`, `synthetic_experiments`, `data_explorations`, 
`model_explorations`, `evaluations`, `sensitivity_analyses`, and `bubble_experiments`.

Documentation
-------------

Build the documentation with

```
tox -e docs
```

Once run, open doc/_build/html/index.html for Sphinx documentation on modules in the package.

Questions and comments
----------------------

Please contact the author at `sgr45 [at] cornell [dot] edu` if you have questions or find bugs.
