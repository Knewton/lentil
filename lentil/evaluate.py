"""
Module for skill model evaluation
@author Siddharth Reddy <sgr45@cornell.edu>
01/09/15
"""

from __future__ import division

import click
import time
import random
import logging

import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics

from lentil import datatools
from lentil import models
from lentil import est


_logger = logging.getLogger(__name__)


def cross_validated_auc(
    model_builders,
    history,
    num_folds=10,
    random_truncations=False):
    """
    Use k-fold cross-validation evaluate
    the predictive power of an embedding model
    on an interaction history

    :param dict[str, function] model_builders:
        A dictionary that maps the name of a model to a function
        that builds and trains it::

        (history, filter_history, init_params, init_model) -> SkillModel

    :param InteractionHistory history: An interaction history

    :param int num_folds:
        Number of folds to make training-validation splits on.
        1 < num_folds <= num_students_in_history

    :param bool random_truncations:
        True => truncate student histories at random locations
        False => truncate student histories just before last batch of assessment interactions

    :rtype: dict[str, (float, float)]
    :return:
        A dictionary mapping model name to a tuple of
        (training roc auc, validation roc auc)
    """
    if num_folds <= 1:
        raise ValueError('Too few folds!')
    num_students = history.num_students()
    if num_folds > num_students:
        raise ValueError('Too many folds!')

    # initialize persistent variables
    df = history.data

    models = {k: None for k in model_builders}

    # predicted pass likelihoods (training set)
    train_probas_pred = {k: [] for k in models}
    # true outcomes (training set)
    train_y_true = []

    # predicted pass likelihoods (validation set)
    val_probas_pred = {k: [] for k in models}
    # true outcomes (validation set)
    val_y_true = []

    # collect errors across CV runs
    err = {k: [] for k in models}

    # define useful helper functions
    def get_training_set(left_out_student_ids):
        """
        Carve out training set by truncating the histories of left-out students

        :param set[str] left_out_student_ids: Left-out students
        :rtype: (pd.Series, pd.DataFrame, tuple, pd.DataFrame, set[str])
        :return: Useful transformations of the training set
        """

        # prepare for left-out student history truncations
        not_in_beginning = df['timestep'] > 2
        is_assessment_ixn = df['module_type'] == datatools.AssessmentInteraction.MODULETYPE
        left_out = df['student_id'].apply(lambda x: x in left_out_student_ids)
        grouped = df[not_in_beginning & is_assessment_ixn & left_out].groupby('student_id')

        if len(grouped) < len(left_out_student_ids):
            # at least one student has no assessment ixns after the second timestep
            raise ValueError(
                'Need to filter out students with too few interactions!')

        if random_truncations:
            # truncate student history at random location
            student_cut_loc = grouped.timestep.apply(np.random.choice) - 1
        else:
            # truncate just before the last batch of assessment ixns for each student
            student_cut_loc = grouped.timestep.max() - 1

        # get timesteps where left-out student histories get truncated
        student_cut_loc.name = 'student_cut_loc'
        truncations = df.join(
            student_cut_loc, on='student_id')['student_cut_loc'].fillna(
            np.nan, inplace=False)

        # get training set, which consists of full student histories
        # for "left-in" students, and truncated histories for "left-out" students
        left_in = np.isnan(truncations)
        left_out = np.logical_not(left_in)
        filtered_history = df[(left_in) | ((left_out) & ((
            df['timestep'] <= truncations) | ((df['timestep'] == truncations+1) & (
                df['module_type'] == datatools.LessonInteraction.MODULETYPE))))]

        # split training set into assessment ixns and lesson ixns
        split_history = history.split_interactions_by_type(
            filtered_history=filtered_history)

        # get assessment ixns in training set
        train_assessment_interactions = filtered_history[(
            filtered_history['module_type'])==(
            datatools.AssessmentInteraction.MODULETYPE)]

        # get set of unique assessment modules in training set
        training_assessments = set(
            train_assessment_interactions['module_id'].values)

        return (
            truncations,
            filtered_history,
            split_history,
            train_assessment_interactions,
            training_assessments)

    def train_models(
        filtered_history,
        split_history):
        """
        Train models on training set

        :param pd.DataFrame filtered_history:
        Interaction history after student truncations

        :param (list, list) split_history:
        A tuple of assessment ixns and lesson ixns
        """

        for k, build_model in model_builders.iteritems():
            _logger.info('Training %s model...', k)
            models[k] = build_model(
                history,
                filtered_history,
                split_history=split_history)

    def collect_labels_and_predictions(
        train_assessment_interactions,
        val_interactions):
        """
        Collect true labels and predicted probabilities

        :param pd.DataFrame train_assessment_interactions:
        Assessment ixns in training set

        :param pd.DataFrame val_interactions: Assessment ixns in validation set
        :rtype: (list[{1,-1}], list[float], list[{1,-1}], list[float])
        :return: (true labels for training set,
        predicted probabilities for training set,
        true labels for validation set,
        predicted probabilities for validation set)
        """

        train_y_true = list(
            train_assessment_interactions['outcome'].apply(
                lambda outcome: 1 if outcome else -1))

        val_y_true = list(
            val_interactions['outcome'].apply(
                lambda outcome: 1 if outcome else -1))

        for k, model in models.iteritems():
            _logger.info('Evaluating %s model...', k)

            train_probas_pred[k] = model.assessment_pass_likelihoods(
                train_assessment_interactions)

            val_probas_pred[k] = model.assessment_pass_likelihoods(
                val_interactions)

        return (train_y_true, train_probas_pred, val_y_true, val_probas_pred)

    def update_err(err):
        """
        Add the current fold's training and validation AUCs to err.
        This function is called at the end of each cross-validation run.

        :param dict[str->(list[float], list[float])] err:
            A dictionary that maps model name to lists
            of training and validation AUCs across folds

        :rtype: dict
        :return: The input parameter err,
            with new training and validation AUCs
            appended to its lists
        """

        for k, v in train_probas_pred.iteritems():
            y_true = [x for x, y in zip(train_y_true, v) if not np.isnan(y)]
            probas_pred = [y for y in v if not np.isnan(y)]
            if len(y_true) == 1:
                raise ValueError('Tried computing AUC with only one prediction!')

            train_fpr, train_tpr, _ = metrics.roc_curve(y_true, probas_pred)
            train_roc_auc = metrics.auc(train_fpr, train_tpr)

            y_true = [x for x, y in zip(val_y_true, val_probas_pred[k]) if not np.isnan(y)]
            probas_pred = [y for y in val_probas_pred[k] if not np.isnan(y)]
            if len(y_true) == 1:
                raise ValueError('Tried computing AUC with only one prediction!')

            val_fpr, val_tpr, _ = metrics.roc_curve(y_true, probas_pred)

            try:
                val_roc_auc = metrics.auc(val_fpr, val_tpr)
            except ValueError:
                # triggered when model doesn't predict any positive labels,
                # i.e., pass outcomes
                val_roc_auc = 0

            _logger.debug("%f %f %s" % (train_roc_auc, val_roc_auc, k))

            err[k].append((train_roc_auc, val_roc_auc))

        return err

    # perform the cross-validated evaluation
    kf = cross_validation.KFold(num_students, n_folds=num_folds, shuffle=True)

    start_time = time.time()

    for fold_idx, (_, val_student_idxes) in enumerate(kf):
        _logger.info('Processing fold %d of %d', fold_idx+1, num_folds)

        left_out_student_ids = {history.id_of_student_idx(
            int(student_idx)) for student_idx in val_student_idxes}

        (
            truncations, filtered_history, split_history,
            train_assessment_interactions,
            training_assessments) = get_training_set(left_out_student_ids)

        train_models(
            filtered_history,
            split_history)

        # validation interactions = assessment interactions that occur
        # immediately after the truncated histories of left-out students
        left_out = np.logical_not(np.isnan(truncations))
        is_train_assessment = df['module_id'].apply(lambda x: x in training_assessments)
        val_interactions = df[(left_out) & (is_train_assessment) & (df['timestep']==truncations+1)]

        (
            train_y_true, train_probas_pred,
            val_y_true, val_probas_pred) = collect_labels_and_predictions(
            train_assessment_interactions,
            val_interactions)

        err = update_err(err)

        _logger.info('Running at %f seconds per iteration',
            (time.time() - start_time) / (fold_idx+1))

    return err

@click.command()
@click.argument('history_file', type=click.Path(exists=True))
@click.option('--verbose', is_flag=True)
@click.option('--num-folds', default=10)
@click.option(
    '--truncation-style',
    type=click.Choice(['random', 'last']),
    default='last')
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
    This script provides a command-line interface for model training.
    It reads an interaction history from file, trains an embedding model,
    and writes the model to file.

    :param str history_file: Input path to CSV file containing interaction history
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

    if verbose:
        _logger.setLevel(logging.DEBUG)

    click.echo('Loading interaction history from %s...' %(
        click.format_filename(history_file)))

    data = pd.DataFrame.from_csv(history_file)
    history = datatools.InteractionHistory(data)

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
        'model' : (
            lambda *args, **kwargs: build_embedding(
                embedding_kwargs,
                estimator,
                *args,
                **kwargs))
        }

    click.echo(
        'Computing cross-validated AUC (num_folds=%d, truncation_style=%s)...' % (
            num_folds,
            truncation_style))

    err = cross_validated_auc(
        model_builders,
        history,
        num_folds=num_folds,
        random_truncations=(truncation_style == 'random'))

    train_aucs = [x[0] for x in err['model']]
    val_aucs = [x[1] for x in err['model']]

    train_auc_avg = np.mean(train_aucs)
    val_auc_avg = np.mean(val_aucs)

    train_auc_stderr = np.std(train_aucs) / len(train_aucs)
    val_auc_stderr = np.std(val_aucs) / len(val_aucs)

    click.echo('AUCs with 95% confidence intervals:')
    click.echo('Training AUC = %f (%f, %f)' % (
        train_auc_avg,
        train_auc_avg - 1.96 * train_auc_stderr,
        train_auc_avg + 1.96 * train_auc_stderr))

    click.echo('Validation AUC = %f (%f, %f)' % (
        val_auc_avg,
        val_auc_avg - 1.96 * val_auc_stderr,
        val_auc_avg + 1.96 * val_auc_stderr))

if __name__ == '__main__':
    cli()
