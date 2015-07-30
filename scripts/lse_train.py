"""
Command-line interface for model training

@author Siddharth Reddy <sgr45@cornell.edu>
"""

import click
import logging
import pickle
import os

import pandas as pd

from lentil import models
from lentil import datatools
from lentil import est
from lentil import evaluate


_logger = logging.getLogger(__name__)


@click.command()
# Path to interaction history CSV/pickle input file
@click.argument('history_file', type=click.Path(exists=True))
# Path to pickle file where trained model should be dumped
@click.argument('model_file', type=click.Path(exists=False))
@click.option(
    '--verbose', is_flag=True,
    help='Makes debug messages visible')
@click.option(
    '--compute-training-auc', is_flag=True, 
    help='Compute training AUC of estimated model')
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
def cli(
    history_file,
    model_file,
    verbose,
    compute_training_auc,
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

    :param str history_file: Input path to CSV/pickle file containing interaction history
    :param str model_file: Output path to pickle file containing trained model
    :param bool verbose: True => logger level set to logging.INFO
    :param bool compute_training_auc: True => compute training AUC of model
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

    click.echo('Loading interaction history from %s...' % (
        click.format_filename(history_file)))

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

    click.echo('Trained model written to %s' % click.format_filename(model_file))

    if compute_training_auc:
        click.echo('Training AUC = %f' % evaluate.training_auc(
            model, history, plot_roc_curve=False))

if __name__ == '__main__':
    cli()

