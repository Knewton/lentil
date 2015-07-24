"""
Module for command-line interface for model training
@author Siddharth Reddy <sgr45@cornell.edu>
01/09/15
"""

import click
import logging
import pickle

import pandas as pd

from lentil import models
from lentil import datatools
from lentil import est


_logger = logging.getLogger(__name__)


@click.command()
@click.argument('history_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path(exists=False))
@click.option('--verbose', is_flag=True)
@click.option('--using-lessons/--no-using-lessons', default=True)
@click.option('--using-prereqs/--no-using-prereqs', default=True)
@click.option('--using-bias/--no-using-bias', default=True)
@click.option('--embedding-dimension', default=2)
@click.option('--learning-update-variance', default=0.5)
@click.option(
    '--opt-algo',
    type=click.Choice(['l-bfgs-b', 'batch-gd', 'adagrad']),
    default='l-bfgs-b')
@click.option('--regularization-constant', default=1e-6)
@click.option('--ftol', default=1e-3)
@click.option('--learning-rate', default=5e-3)
@click.option('--adagrad-eta', default=1e-3)
@click.option('--adagrad-eps', default=0.1)
def cli(
    history_file,
    model_file,
    verbose,
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
    This script provides a command-line interface for model training.
    It reads an interaction history from file, trains an embedding model,
    and writes the model to file.

    :param str history_file: Input path to CSV file containing interaction history
    :param str model_file: Output path to pickle file containing trained model
    :param bool verbose: True => logger level set to logging.INFO
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

    if verbose:
        _logger.setLevel(logging.DEBUG)

    click.echo('Loading interaction history from %s...' % (
        click.format_filename(history_file)))

    data = pd.DataFrame.from_csv(history_file)
    history = datatools.InteractionHistory(data)

    click.echo('Computing MAP estimates of model parameters...')

    model = models.EmbeddingModel(
        history,
        embedding_dimension,
        using_lessons=using_lessons,
        using_prereqs=using_prereqs,
        using_bias=using_bias)

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

    model.fit(estimator)

    with open(model_file, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    click.echo('Trained model written to %s' % (
        click.format_filename(model_file)))

if __name__ == '__main__':
    cli()
