"""
Command-line interface for model evaluation

@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import click
import logging
import math
import pickle
import os

import pandas as pd
import numpy as np

from lentil import datatools
from lentil import models
from lentil import est
from lentil import evaluate


_logger = logging.getLogger(__name__)


@click.command()
# path to interaction history CSV/pickle input file
@click.argument('history_file', type=click.Path(exists=True))
# path to pickled results file
@click.argument('results_file', type=click.Path(exists=False))
@click.option(
    '--verbose', is_flag=True,
    help='Makes debug messages visible')
@click.option(
    '--using-lessons/--no-using-lessons', default=True,
    help='Include embeddings of skill gains from lessons')
@click.option(
    '--using-prereqs/--no-using-prereqs', default=True,
    help='Include embeddings of prerequisites for lessons')
@click.option(
    '--using-bias/--no-using-bias', default=True,
    help='Include bias terms in the item response function')
@click.option(
    '--embedding-dimension', default=2,
    help='Dimensionality of latent skill space')
@click.option(
    '--learning-update-variance', default=0.5,
    help='Constant variance for Gaussian lesson updates')
@click.option(
    '--opt-algo',
    type=click.Choice(['l-bfgs-b', 'batch-gd', 'adagrad']),
    default='l-bfgs-b',
    help='Iterative optimization algorithm used for parameter estimation')
@click.option(
    '--regularization-constant', default=1e-6,
    help='Coefficient of norm regularization terms')
@click.option(
    '--ftol', default=1e-3,
    help='Stopping condition for iterative optimization')
@click.option('--learning-rate', default=5e-3, help='Fixed learning rate')
@click.option('--adagrad-eta', default=1e-3, help='Adagrad learning rate')
@click.option('--adagrad-eps', default=0.1, help='Adagrad epsilon')
@click.option('--num-folds', default=10, help='Number of folds in k-fold cross-validation')
@click.option(
    '--truncation-style',
    type=click.Choice(['random', 'last']),
    default='last',
    help='Truncate student history at random, or just before last assessment interactions')
def cli(
    history_file,
    results_file,
    verbose,
    num_folds,
    truncation_style,
    using_lessons,
    using_prereqs,
    using_bias,
    embedding_dimension,
    learning_update_variance,
    opt_algo,
    regularization_constant,
    ftol,
    learning_rate,
    adagrad_eta,
    adagrad_eps):
    """
    This script provides a command-line interface for model evaluation.
    It reads an interaction history from file, computes the cross-validated AUC of
    an embedding model, and writes the results to file.

    The pickled results will be an object of type :py:class:`evaluate.CVResults`

    :param str history_file: Input path to CSV/pickle file containing interaction history
    :param str results_file: Output path for pickled results of cross-validation
    :param bool verbose: True => logger level set to logging.INFO
    :param int num_folds: Number of folds in k-fold cross-validation
    :param str truncation_style: Hold-out scheme for student histories
    :param bool using_lessons: Including lessons in embedding
    :param bool using_prereqs: Including lesson prereqs in embedding
    :param bool using_bias: Including bias terms in embedding
    :param int embedding_dimension: Number of dimensions in latent skill space
    :param float learning_update_variance: Variance of Gaussian learning update
    :param str opt_algo: Optimization algorithm for parameter estimation
    :param float regularization_constant: Coefficient of regularization term in objective function
    :param float ftol: Stopping condition for iterative optimization
    :param float learning_rate: Fixed learning rate for gradient descent
    :param float adagrad_eta: Base learning rate parameter for Adagrad
    :param float adagrad_eps: Epsilon parameter for Adagrad
    """

    if verbose and opt_algo == 'l-bfgs-b':
        raise ValueError('Verbose mode is not currently supported for L-BFGS-B.\
                Try turning off verbose mode, or change your choice of optimization algorithm.')

    if verbose:
        _logger.setLevel(logging.DEBUG)

    click.echo('Loading interaction history from %s...' % click.format_filename(history_file))

    _, history_file_ext = os.path.splitext(history_file)
    if history_file_ext == '.csv':
        data = pd.DataFrame.from_csv(history_file)
        history = datatools.InteractionHistory(pd.read_csv(history_file))
    elif history_file_ext == '.pkl':
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
    else:
        raise ValueError('Unrecognized file extension for history_file.\
                Please supply a .csv with an interaction history, or a .pkl file containing\
                a datatools.InteractionHistory object.')

    embedding_kwargs = {
        'embedding_dimension' : embedding_dimension,
        'using_lessons' : using_lessons,
        'using_prereqs' : using_prereqs,
        'using_bias' : using_bias,
        'learning_update_variance_constant' : learning_update_variance
    }

    gradient_descent_kwargs = {
        'using_adagrad' : opt_algo == 'adagrad',
        'eta' : adagrad_eta,
        'eps' : adagrad_eps,
        'rate' : learning_rate,
        'verify_gradient' : False,
        'debug_mode_on' : verbose,
        'ftol' : ftol,
        'num_checkpoints' : 100
    }

    estimator = est.EmbeddingMAPEstimator(
        regularization_constant=regularization_constant,
        using_scipy=(opt_algo == 'l-bfgs-b'),
        gradient_descent_kwargs=gradient_descent_kwargs,
        verify_gradient=False,
        debug_mode_on=verbose,
        ftol=ftol)

    def build_embedding(
        embedding_kwargs,
        estimator,
        history,
        filtered_history,
        split_history=None):

        model = models.EmbeddingModel(history, **embedding_kwargs)

        estimator.filtered_history = filtered_history
        if split_history is not None:
            estimator.split_history = split_history

        model.fit(estimator)

        return model

    model_builders = {
        'model' : (lambda *args, **kwargs: build_embedding(
            embedding_kwargs,
            estimator,
            *args,
            **kwargs))
        }

    click.echo(
        'Computing cross-validated AUC (num_folds=%d, truncation_style=%s)...' % (
            num_folds,
            truncation_style))

    results = evaluate.cross_validated_auc(
        model_builders,
        history,
        num_folds=num_folds,
        random_truncations=(truncation_style == 'random'))

    train_auc_mean = results.training_auc_mean('model')
    val_auc_mean = results.validation_auc_mean('model')

    train_auc_stderr = results.training_auc_stderr('model')
    val_auc_stderr = results.validation_auc_stderr('model')

    click.echo('AUCs with 95% confidence intervals:')
    click.echo('Training AUC = %f (%f, %f)' % (
        train_auc_mean,
        train_auc_mean - 1.96 * train_auc_stderr,
        train_auc_mean + 1.96 * train_auc_stderr))

    click.echo('Validation AUC = %f (%f, %f)' % (
        val_auc_mean,
        val_auc_mean - 1.96 * val_auc_stderr,
        val_auc_mean + 1.96 * val_auc_stderr))

    with open(results_file, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    click.echo('Results written to %s' % results_file)

if __name__ == '__main__':
    cli()

