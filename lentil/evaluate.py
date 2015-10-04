"""
Module for skill model evaluation

@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import logging
import random
import time
import math

from matplotlib import pyplot as plt
from sklearn import cross_validation, metrics
from scipy import stats
import numpy as np

from . import datatools


_logger = logging.getLogger(__name__)

# students with fewer than [MIN_NUM_TIMESTEPS_IN_STUDENT_HISTORY] timesteps
# in their history need to be filtered out before calling cross_validated_auc
MIN_NUM_TIMESTEPS_IN_STUDENT_HISTORY = 2


class EvalResults(object):
    """
    Class for wrapping the results of evaluation on the assessment outcome prediction task
    """

    def __init__(self, raw_results, raw_test_results=None):
        """
        Initialize results object

        :param dict[str,list[(float,float,float,float)]] raw_results: A dictionary mapping model name to
            a list of tuples (training AUC, validation AUC, validation accuracy, 
            stdev of validation accuracy) across CV runs

        :param dict[str,(float,float,float,float)]|None raw_test_results: A dictionary mapping model name to
            a tuple (training AUC, test AUC, test accuracy, stdev of test accuracy)
        """
        self.raw_results = raw_results
        self.raw_test_results = raw_test_results if raw_test_results is not None else {}

    def training_aucs(self, model):
        """
        Get training AUCs across CV runs

        :param str model: The name of a model
        :rtype: np.array
        :return: Training AUCs
        """

        train_aucs = np.array([t[0] for t in self.raw_results[model] if t is not None])
        return train_aucs[~np.isnan(train_aucs)]

    def validation_aucs(self, model):
        """
        Get validation AUCs across CV runs

        :param str model: The name of a model
        :rtype: np.array
        :return: Validation AUCs
        """

        val_aucs = np.array([t[1] for t in self.raw_results[model] if t is not None])
        return val_aucs[~np.isnan(val_aucs)]
    
    def training_auc_mean(self, model):
        """
        Compute mean training AUC across CV runs

        :param str model: The name of a model
        :rtype: float
        :return: Mean training AUC
        """

        return np.mean(self.training_aucs(model))

    def validation_auc_mean(self, model):
        """
        Compute mean validation AUC across CV runs

        :param str model: The name of a model
        :rtype: float
        :return: Mean validation AUC
        """

        return np.mean(self.validation_aucs(model))

    def training_auc_stderr(self, model):
        """
        Compute standard error of training AUC across CV runs

        :param str model: The name of a model
        :rtype: float
        :return: Standard error of training AUC
        """

        train_aucs = self.training_aucs(model)
        return np.std(train_aucs) / math.sqrt(len(train_aucs))

    def validation_auc_stderr(self, model):
        """
        Compute standard error of validation AUC across CV runs

        :param str model: The name of a model
        :rtype: float
        :return: Standard error of validation AUC
        """

        val_aucs = self.validation_aucs(model)
        return np.std(val_aucs) / math.sqrt(len(val_aucs))

    def compare_validation_aucs(self, model_a, model_b):
        """
        Use a paired t-test to check the statistical significance
        of the difference between the validation AUCs (across CV runs) of two models

        :param str model_a: The name of a model
        :param str model_b: The name of another model
        :rtype: float
        :return: p-value
        """

        return stats.ttest_ind(
                self.validation_aucs(model_a), self.validation_aucs(model_b), equal_var=True)[1]

    def test_auc(self, model):
        """
        Get the test AUC of a model

        :param str model: The name of a model
        :rtype float|None
        :return: The test AUC of the model, or None if test results were not supplied
        """

        return self.raw_test_results[model][1] if model in self.raw_test_results else None

    def test_acc(self, model):
        """
        Get the test accuracy of a model

        :param str model: The name of a model
        :rtype float|None
        :return: The test accuracy of the model, or None if test results were not supplied
        """

        return self.raw_test_results[model][2] if model in self.raw_test_results else None

    def test_acc_stderr(self, model):
        """
        Get the standard error of the test accuracy of a model

        :param str model: The name of a model
        :rtype float|None
        :return: The standard error of the test accuracy of the model, or None if the test results
            were not supplied
        """

        return self.raw_test_results[model][3] if model in self.raw_test_results else None

    def merge(self, other_results):
        """
        Merge another results object with self (not in-place)

        :param EvalResults other_results: A results object
        :rtype: EvalResults
        :return: A combined results object
        """

        combined_raw_results = self.raw_results
        combined_raw_results.update(other_results.raw_results)
        combined_raw_test_results = self.raw_test_results
        combined_raw_test_results.update(other_results.raw_test_results)
        return EvalResults(combined_raw_results)

def training_auc(
    model, 
    history,
    plot_roc_curve=True):
    """
    Compute the training AUC of a trained model on an interaction history

    :param models.SkillModel model: A trained model
    :param datatools.InteractionHistory history: The interaction history used to train the model
    :param bool plot_roc_curve: True => plot ROC curve
    :rtype: float|None
    :return: Area under ROC curve
    """

    train_assessment_interactions = history.data[history.data['module_type'] == \
            datatools.AssessmentInteraction.MODULETYPE]
    train_y_true = train_assessment_interactions['outcome'] * 2 - 1
    train_probas_pred = model.assessment_pass_likelihoods(train_assessment_interactions)
    y_true = [x for x, y in zip(train_y_true, train_probas_pred) if not np.isnan(y)]
    probas_pred = [y for y in train_probas_pred if not np.isnan(y)]
    if len(y_true) == 1:
        raise ValueError('Tried computing AUC with only one prediction!')
    
    try:
        train_fpr, train_tpr, _ = metrics.roc_curve(y_true, probas_pred)
        train_roc_auc = metrics.auc(train_fpr, train_tpr)
    except:
        _logger.debug('Could not compute training AUC for y_true and probas_pred:')
        _logger.debug(y_true)
        _logger.debug(probas_pred)
        return None

    if plot_roc_curve:
        _, ax = plt.subplots()
        ax.plot(train_fpr, train_tpr, label='ROC curve (area = %0.2f)' % train_roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        plt.show()

    return train_roc_auc


def cross_validated_auc(
    model_builders,
    history,
    num_folds=10,
    random_truncations=False,
    size_of_test_set=0.2):
    """
    Use k-fold cross-validation to evaluate the predictive power of an
    embedding model on an interaction history

    Each student history needs to be longer than three timesteps, and doesn't necessarily have
    to contain lesson interactions

    :param dict[str,function] model_builders:
        A dictionary that maps the name of a model to a function
        that builds and trains it::

        (datatools.InteractionHistory, pd.DataFrame, datatools.SplitHistory) -> models.SkillModel

        See nb/evaluations.ipynb for examples

    :param datatools.InteractionHistory history: An interaction history

    :param int num_folds:
        Number of folds to make training-validation splits on.
        1 < num_folds <= num_students_in_history

    :param bool random_truncations:
        True => truncate student histories at random locations
        False => truncate student histories just before last batch of assessment interactions

    :param float size_of_test_set: Fraction of students to include in the test set, where
        0 <= size_of_test_set < 1 (size_of_test_set = 0 => don't compute test AUCs)

    :rtype: dict[str,(float,float)]
    :return:
        A dictionary mapping model name to a tuple of (training roc auc, validation roc auc)
    """
    if num_folds <= 1:
        raise ValueError('Too few folds! Must be at least 1 not {}'.format(num_folds))
    num_students = history.num_students()
    if num_folds > num_students:
        raise ValueError('Too many folds! Must be at most num_students ({}) not {}'.format(
                         num_students, num_folds))

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
    def get_training_and_validation_sets(left_in_student_ids, left_out_student_ids):
        """
        Carve out training/validation sets by truncating the histories of left-out students

        :param set[str] left_in_student_ids: Left-in students
        :param set[str] left_out_student_ids: Left-out students
        :rtype: (pd.DataFrame,pd.DataFrame,datatools.SplitHistory,pd.DataFrame)
        :return:
            (assessment interactions in training set,
            interactions in training set,
            training set split into assessment/lesson ixns and timestep_of_last_interaction,
            interactions in validation set)
        """

        # prepare for left-out student history truncations
        not_in_beginning = df['timestep'] > MIN_NUM_TIMESTEPS_IN_STUDENT_HISTORY
        is_assessment_ixn = df['module_type'] == datatools.AssessmentInteraction.MODULETYPE
        left_out = df['student_id'].isin(left_out_student_ids)
        grouped = df[not_in_beginning & is_assessment_ixn & left_out].groupby('student_id')

        if len(grouped) < len(left_out_student_ids):
            # at least one student has no assessment ixns after the second timestep
            raise ValueError('Need to filter out students with too few interactions!')

        if random_truncations:
            # truncate student history at random location
            # after timestep [MIN_NUM_TIMESTEPS_IN_STUDENT_HISTORY]
            student_cut_loc = grouped.timestep.apply(lambda x: np.maximum(
                MIN_NUM_TIMESTEPS_IN_STUDENT_HISTORY, np.random.choice(x))) - 1
        else:
            # truncate just before the last batch of assessment ixns for each student
            student_cut_loc = grouped.timestep.max() - 1

        # get timesteps where left-out student histories get truncated
        student_cut_loc.name = 'student_cut_loc'
        truncations = df.join(
            student_cut_loc, on='student_id')['student_cut_loc'].fillna(np.nan, inplace=False)

        # get training set, which consists of full student histories
        # for "left-in" students, and truncated histories for "left-out" students
        left_in = df['student_id'].isin(left_in_student_ids)
        filtered_history = df[left_in | (left_out & ((
            df['timestep'] <= truncations) | ((df['timestep'] == truncations+1) & (
                df['module_type'] == datatools.LessonInteraction.MODULETYPE))))]

        # split training set into assessment ixns and lesson ixns
        split_history = history.split_interactions_by_type(
                filtered_history=filtered_history,
                insert_dummy_lesson_ixns=False)

        # get assessment ixns in training set
        train_assessment_interactions = filtered_history[
            filtered_history['module_type']==datatools.AssessmentInteraction.MODULETYPE]

        # get set of unique assessment modules in training set
        training_assessments = set(train_assessment_interactions['module_id'].values)

        # validation interactions = assessment interactions that occur
        # immediately after the truncated histories of left-out students
        module_in_train_set = df['module_id'].isin(training_assessments)
        val_interactions = df[left_out & module_in_train_set & (
            df['timestep']==truncations+1) & is_assessment_ixn]
      
        return (train_assessment_interactions, filtered_history, split_history, val_interactions)

    def train_models(
        filtered_history,
        split_history):
        """
        Train models on training set

        :param pd.DataFrame filtered_history: Interaction history after student truncations
        :param datatools.SplitHistory split_history: An interaction history split into
            assessment interactions, lesson interactions, and timestep of last interaction
            for each student
        """

        for k, build_model in model_builders.iteritems():
            _logger.info('Training %s model...', k)
            models[k] = build_model(history, filtered_history, split_history=split_history)

    def collect_labels_and_predictions(
        train_assessment_interactions,
        val_interactions):
        """
        Collect true labels and predicted probabilities

        :param pd.DataFrame train_assessment_interactions: Assessment ixns in training set
        :param pd.DataFrame val_interactions: Assessment ixns in validation set
        :rtype: (list[{1,-1}],list[float],list[{1,-1}],list[float])
        :return: (true labels for training set,
                  predicted probabilities for training set,
                  true labels for validation set,
                  predicted probabilities for validation set)
        """

        train_y_true = (2 * train_assessment_interactions['outcome'] - 1).values
        val_y_true = (2 * val_interactions['outcome'] - 1).values

        for k, model in models.iteritems():
            _logger.info('Evaluating %s model...', k)
            train_probas_pred[k] = model.assessment_pass_likelihoods(train_assessment_interactions)
            val_probas_pred[k] = model.assessment_pass_likelihoods(val_interactions)

        return (train_y_true, train_probas_pred, val_y_true, val_probas_pred)

    def update_err(err):
        """
        Add the current fold's training and validation AUCs to err.
        This function is called at the end of each cross-validation run.

        :param dict[str,list[(float,float)]] err:
            A dictionary that maps model name to a list of training/validation AUCs across folds

        :rtype: dict[str,list[(float,float)]]
        :return: The input parameter err, with new training and validation AUCs
            appended to its lists
        """

        for k, v in train_probas_pred.iteritems():
            y_true = [x for x, y in zip(train_y_true, v) if not np.isnan(y)]
            probas_pred = [y for y in v if not np.isnan(y)]
            if len(y_true) == 1:
                raise ValueError('Tried computing AUC with only one prediction!')

            try:
                train_fpr, train_tpr, _ = metrics.roc_curve(y_true, probas_pred)
                train_roc_auc = metrics.auc(train_fpr, train_tpr)
            except:
                _logger.debug('Could not compute training AUC for y_true and probas_pred:')
                _logger.debug(y_true)
                _logger.debug(probas_pred)
                train_roc_auc = None

            y_true = [x for x, y in zip(val_y_true, val_probas_pred[k]) if not np.isnan(y)]
            probas_pred = [y for y in val_probas_pred[k] if not np.isnan(y)]
            if len(y_true) == 1:
                raise ValueError('Tried computing AUC with only one prediction!')

            try:
                val_fpr, val_tpr, val_thresholds = metrics.roc_curve(y_true, probas_pred)
                val_roc_auc = metrics.auc(val_fpr, val_tpr)
            except:
                _logger.debug('Could not compute validation AUC for y_true and probas_pred:')
                _logger.debug(y_true)
                _logger.debug(probas_pred)
                val_roc_auc = None
            
            y_pred = [1 if x>=0.5 else -1 for x in probas_pred]
            val_acc = np.array([1 if p==t else 0 for p, t in zip(y_pred, y_true)])

            # helpful if you want to do a sanity check on AUCs
            # but don't want to wait for all folds to finish running
            _logger.debug('Model = %s', k)
            _logger.debug('Training AUC = %f', train_roc_auc)
            _logger.debug('Validation AUC = %f', val_roc_auc)
            _logger.debug('Validation Accuracy = %f +/- %f', np.mean(val_acc), 
                    np.std(val_acc) / np.sqrt(len(val_acc)))

            err[k].append((train_roc_auc, val_roc_auc, np.mean(val_acc), 
                np.std(val_acc) / np.sqrt(len(val_acc))))

        return err

    # make train-test splits for CV runs
    kf = cross_validation.KFold(
            num_students - len(history.id_of_nontest_student_idx), n_folds=num_folds, shuffle=True)

    start_time = time.time()

    for fold_idx, (train_student_idxes, val_student_idxes) in enumerate(kf):
        _logger.info('Processing fold %d of %d', fold_idx+1, num_folds)

        left_in_student_ids = {history.id_of_nontest_student_idx[student_idx] \
                for student_idx in train_student_idxes}
        left_out_student_ids = {history.id_of_nontest_student_idx[student_idx] \
                for student_idx in val_student_idxes}

        train_assessment_interactions, filtered_history, split_history, val_interactions = \
                get_training_and_validation_sets(left_in_student_ids, left_out_student_ids)

        train_models(filtered_history, split_history)

        train_y_true, train_probas_pred, val_y_true, val_probas_pred = \
                collect_labels_and_predictions(train_assessment_interactions, val_interactions)

        err = update_err(err)

        _logger.info('Running at %f seconds per fold', (time.time() - start_time) / (fold_idx+1))

    if size_of_test_set > 0:
        _logger.info('Computing test AUCs...')
        all_student_ids = set(history._student_idx.keys())
        nontest_student_ids = set(history.id_of_nontest_student_idx.values())
        train_assessment_interactions, filtered_history, split_history, val_interactions = \
                get_training_and_validation_sets(
                        nontest_student_ids, all_student_ids - nontest_student_ids)

        train_models(filtered_history, split_history)

        train_y_true, train_probas_pred, val_y_true, val_probas_pred = \
                collect_labels_and_predictions(train_assessment_interactions, val_interactions)

        err = update_err(err)

        test_err = {k: v[-1] for k, v in err.iteritems()}
        err = {k: v[:-1] for k, v in err.iteritems()}
    else:
        test_err = None
    

    return EvalResults(err, raw_test_results=test_err)

