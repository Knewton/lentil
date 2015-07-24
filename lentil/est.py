"""
Module for estimating model parameters
@author Siddharth Reddy <sgr45@cornell.edu>
01/07/15
"""

from __future__ import division

from abc import abstractmethod
from collections import OrderedDict
import math
import copy
import sys
import time
import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import sparse

from lentil import models
from lentil import grad


_logger = logging.getLogger(__name__)


def gradient_descent(
    grads,
    params,
    param_constraint_funcs,
    eta=0.001,
    eps=0.1,
    rate=0.005,
    ftol=1e-3,
    num_checkpoints=10,
    using_adagrad=True,
    debug_mode_on=False,
    verify_gradient=False):
    """
    Batch gradient descent

    Optionally uses Adagrad to compute adaptive learning rate

    :param function grads: A function that takes parameter values,
        assessment interactions, and lesson interactions as input,
        and outputs values for gradients and the cost function
        evaluated with current parameter values

    :param dict[str, np.ndarray] params: Parameters allowed to vary

    :param dict[str, function] param_constraint_funcs:
        Functions that enforce bounds on parameters

    :param float eta: Adagrad parameter
    :param float eps: Adagrad parameter
    :param bool debug_mode_on: True => dump plots
    :param int num_checkpoints: Number of times to print "Iteration #" during updates
    :rtype: dict[str, np.ndarray]
    :return: Parameter values at which gradient descent "converges"
    """
    if eta<=0:
        raise ValueError('Invalid eta!')
    if eps<=0:
        raise ValueError('Invalid eps!')
    if num_checkpoints<=1:
        raise ValueError('Invalid number of checkpoints!')

    if using_adagrad:
        # historical gradient (for Adagrad)
        hg = {k: np.zeros(v.shape) for k, v in params.iteritems()}

    est_training_steps = 1000
    checkpoint_iter_step = max(1, est_training_steps // num_checkpoints)
    is_checkpoint = lambda idx: idx % checkpoint_iter_step == 0

    if verify_gradient:
        # check accuracy of gradient function
        # using finite differences
        g, cst = grads(params)

        epsilon = 1e-9 # epsilon upper bound
        num_samples = 1 # number of epsilon samples to draw
        for k, v in g.iteritems():
            if v is None:
                continue
            fd = [] # gradient components computed with finite difference method
            cf = [] # gradient components computed with closed form expression
            for i, c in enumerate(v.flatten()):
                for _ in xrange(num_samples):
                    nparams = copy.deepcopy(params)
                    delta = epsilon * c

                    nparams[k] = nparams[k].flatten()
                    nparams[k][i] += delta
                    nparams[k] = np.reshape(nparams[k], params[k].shape)
                    _, ncst = grads(nparams)

                    nparams[k] = nparams[k].flatten()
                    nparams[k][i] -= 2 * delta
                    nparams[k] = np.reshape(nparams[k], params[k].shape)
                    _, pcst = grads(nparams)

                    fd.append((ncst - pcst) / (2 * delta))
                cf += [c] * num_samples

            plt.title('Components of Cost Gradient w.r.t. %s' % k)
            plt.xlabel('From Finite Differences')
            plt.ylabel('From Closed Form Expression')
            plt.scatter(fd, cf)
            plt.show()

            try:
                diffs = [(x-y)/x for x, y in zip(fd, cf)]
                plt.title('Diffs of Cost Gradient w.r.t. %s' % k)
                plt.xlabel(
                    '(From Finite Differences - From Closed Form Expression) \
                    / From Finite Differences')
                plt.ylabel('Frequency (number of gradient components)')
                plt.hist([x for x in diffs if not (np.isinf(x) or np.isnan(x))], bins=20)
                plt.show()
            except IndexError:
                pass

    costs = []
    gs = {k: [] for k in params}

    start_time = time.time()

    iter_idx = 0
    rel_diff = sys.maxint
    while True:
        g, cst = grads(params)

        for k in params:
            v = g[k]
            if using_adagrad:
                params[k] -= eta / (eps + np.sqrt(hg[k])) * v
                hg[k] += np.square(v)
            else:
                params[k] -= rate * v

            params[k] = param_constraint_funcs[k](k, params[k])

        costs.append(cst)
        for k in params:
            gs[k].append(np.linalg.norm(g[k]))

        if iter_idx >= 1:
            rel_diff = (cst - costs[-2]) / costs[-2]

        if is_checkpoint(iter_idx):
            _logger.debug('Iteration %d, rel_diff=%f' % (iter_idx, rel_diff))
            _logger.debug('Running at %f seconds per iteration' % (
                (time.time() - start_time) / (iter_idx+1)))

        if abs(rel_diff) <= ftol:
            break

        iter_idx += 1

    if debug_mode_on:
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.plot(costs)
        plt.show()
        for k, v in gs.iteritems():
            plt.ylabel('L2-norm-squared of g'+k)
            plt.plot(v)
            plt.show()

    return params

class EmbeddingMAPEstimator(object):
    """
    Trains a model on an interaction history by computing
    maximum a posteriori estimates of model parameters
    """

    def __init__(
        self,
        regularization_constant=1e-6,
        ftol=1e-3,
        gradient_descent_kwargs={},
        initial_param_vals={},
        using_scipy=True,
        verify_gradient=False,
        debug_mode_on=False,
        filtered_history=None,
        split_history=None):
        """
        Initialize estimator object

        :param float regularization_constant: Coefficient of L2 regularizer
            in cost function

        :param dict[str, object] gradient_descent_kwargs:
            Arguments for gradient_descent

        :param dict[str, np.ndarray] initial_param_vals: For warm starts
        :param bool using_scipy:
            True => use scipy.optimize.minimize,
            False => use batch gradient descent

        :param bool verify_gradient:
            True => use scipy.optimize.check_grad to verify accuracy of analytic gradient,

        :param bool debug_mode_on:
            True => print checkpoints and dump plots
        """
        if regularization_constant<0:
            raise ValueError('Invalid regularizer coefficient!')
        if ftol<=0:
            raise ValueError('Invalid ftol!')

        if type(regularization_constant) is float:
            # same regularization constant for all embedding parameters
            # (students, assessments, lessons, prereqs, and concepts)
            regularization_constant = [regularization_constant] * 5

        self.regularization_constant = regularization_constant
        self.gradient_descent_kwargs = gradient_descent_kwargs
        self.initial_param_vals = initial_param_vals
        self.filtered_history = filtered_history
        self.split_history = split_history
        self.ftol = ftol

        self.using_scipy = using_scipy
        self.verify_gradient = verify_gradient
        self.debug_mode_on = debug_mode_on

    def fit_model(self, model):
        """
        Use gradient descent to perform maximum a posteriori
        estimation of model parameters. Relies on hand-coded gradient.

        :param EmbeddingModel model: A skill embedding model
            that needs to be fit to its interaction history
        """

        param_shapes = OrderedDict([
            (models.STUDENT_EMBEDDINGS, (
                model.student_embeddings.shape[0] * model.student_embeddings.shape[2],
                model.student_embeddings.shape[1])),
            (models.ASSESSMENT_EMBEDDINGS, model.assessment_embeddings.shape)
            ])

        if model.using_lessons:
            param_shapes[models.LESSON_EMBEDDINGS] = model.lesson_embeddings.shape
        if model.using_prereqs:
            param_shapes[models.PREREQ_EMBEDDINGS] = model.prereq_embeddings.shape

        param_shapes[models.STUDENT_BIASES] = model.student_biases.shape
        param_shapes[models.ASSESSMENT_BIASES] = model.assessment_biases.shape

        if model.using_graph_prior:
            param_shapes[models.CONCEPT_EMBEDDINGS] = model.concept_embeddings.shape

        constraint_func = lambda name, val: np.maximum(
            model.anti_singularity_lower_bounds[name],
            val)
        no_constraint_func = lambda name, val: val
        param_constraint_funcs = {
            models.STUDENT_EMBEDDINGS : constraint_func,
            models.ASSESSMENT_EMBEDDINGS : constraint_func,
            models.LESSON_EMBEDDINGS : constraint_func,
            models.PREREQ_EMBEDDINGS : constraint_func,
            models.STUDENT_BIASES : no_constraint_func,
            models.ASSESSMENT_BIASES : no_constraint_func,
            models.CONCEPT_EMBEDDINGS : constraint_func,
        }

        params = OrderedDict((k,
                param_constraint_funcs[k](k,
                    self.initial_param_vals[k]) if k in \
                self.initial_param_vals else param_constraint_funcs[k](k,
                    np.random.random(v))) for k, v in param_shapes.iteritems())

        if self.split_history is None:
            (
                assessment_interactions,
                lesson_interactions,
                timestep_of_last_interaction) = model.history.split_interactions_by_type(
                filtered_history=self.filtered_history)
        else:
            (
                assessment_interactions,
                lesson_interactions,
                timestep_of_last_interaction) = self.split_history

        if assessment_interactions == []:
            raise ValueError('No assessment interactions in history!')

        grads = grad.get_grad(
            using_scipy=self.using_scipy,
            using_lessons=model.using_lessons,
            using_prereqs=model.using_prereqs)

        param_sizes = {k: v.size for k, v in params.iteritems()}
        param_vals = np.concatenate([v.flatten(
            ) for v in params.itervalues(
            )], axis=0)

        (
            student_idxes_for_assessment_ixns,
            assessment_idxes_for_assessment_ixns, _) = assessment_interactions
        if model.using_lessons:
            (
                student_idxes_for_lesson_ixns,
                lesson_idxes_for_lesson_ixns,
                times_since_prev_ixn_for_lesson_ixns) = lesson_interactions
            num_lesson_ixns = len(student_idxes_for_lesson_ixns)
            lesson_ixns_participation_matrix_entries = np.ones(num_lesson_ixns)
        else:
            times_since_prev_ixn_for_lesson_ixns = None

        num_assessment_ixns = len(student_idxes_for_assessment_ixns)
        assessment_ixns_participation_matrix_entries = np.ones(num_assessment_ixns)
        assessment_ixn_idxes = np.arange(num_assessment_ixns)
        num_students = param_shapes[models.STUDENT_EMBEDDINGS][0]
        num_assessments = param_shapes[models.ASSESSMENT_EMBEDDINGS][0]
        assessment_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (assessment_idxes_for_assessment_ixns, assessment_ixn_idxes)),
            shape=(num_assessments, num_assessment_ixns)).tocsr()
        student_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (student_idxes_for_assessment_ixns, assessment_ixn_idxes)),
            shape=(num_students, num_assessment_ixns)).tocsr()

        num_timesteps = model.history.duration()
        student_bias_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (student_idxes_for_assessment_ixns // num_timesteps,
                    assessment_ixn_idxes)),
            shape=(num_students // num_timesteps, num_assessment_ixns)).tocsr()

        if not model.using_graph_prior:
            assessment_participation_in_concepts = None
            lesson_participation_in_concepts = None
            concept_participation_in_assessments = None
            concept_participation_in_lessons = None
        else:
            (
                assessment_idxes,
                concept_idxes,
                num_assessments,
                num_concepts,
                num_concepts_per_assessment) = model.concept_assessment_edges_in_graph()
            assessment_participation_in_concepts = sparse.coo_matrix(
                (1 / num_concepts_per_assessment, (assessment_idxes, concept_idxes)),
                shape=(num_assessments, num_concepts)).tocsr()
            concept_participation_in_assessments = assessment_participation_in_concepts.T

        if model.using_lessons:
            lesson_ixn_idxes = np.arange(num_lesson_ixns)
            curr_student_participation_in_lesson_ixns = sparse.coo_matrix(
                (lesson_ixns_participation_matrix_entries,
                    (student_idxes_for_lesson_ixns, lesson_ixn_idxes)),
                shape=(num_students, num_lesson_ixns)).tocsr()
            prev_student_participation_in_lesson_ixns = sparse.coo_matrix(
                (lesson_ixns_participation_matrix_entries,
                    (student_idxes_for_lesson_ixns - 1, lesson_ixn_idxes)),
                shape=(num_students, num_lesson_ixns)).tocsr()

            if not model.using_prereqs:
                # when computing gradients, we will be computing Ax - Bx
                # where A = curr_student_participation_in_lesson_ixns
                # and B = prev_student_participation_in_lesson_ixns,
                # so to speed things up we precompute A-B and compute (A-B)x
                curr_student_participation_in_lesson_ixns -= (
                    prev_student_participation_in_lesson_ixns)

            num_lessons = param_shapes[models.LESSON_EMBEDDINGS][0]
            lesson_participation_in_lesson_ixns = sparse.coo_matrix(
                (lesson_ixns_participation_matrix_entries,
                    (lesson_idxes_for_lesson_ixns, lesson_ixn_idxes)),
                shape=(num_lessons, num_lesson_ixns)).tocsr()

            if model.using_graph_prior:
                (
                    lesson_idxes,
                    concept_idxes,
                    num_lessons,
                    num_concepts,
                    num_concepts_per_lesson) = model.concept_lesson_edges_in_graph()
                lesson_participation_in_concepts = sparse.coo_matrix(
                    (1/num_concepts_per_lesson, (lesson_idxes, concept_idxes)),
                    shape=(num_lessons, num_concepts))
                concept_participation_in_lessons = lesson_participation_in_concepts.T
        else:
            curr_student_participation_in_lesson_ixns = None
            prev_student_participation_in_lesson_ixns = None
            lesson_participation_in_lesson_ixns = None
            lesson_participation_in_concepts = None
            concept_participation_in_lessons = None

        if model.using_graph_prior:
            prereq_idxes, postreq_idxes, num_concepts = model.graph.concept_prereq_edges()
            entries = np.ones(len(prereq_idxes))
            num_entries = len(entries)
            entry_idxes = np.arange(0, num_entries, 1)
            prereq_edge_concept_idxes = prereq_idxes, postreq_idxes
            concept_participation_in_prereqs = sparse.coo_matrix(
                (entries, (prereq_idxes, entry_idxes)),
                shape=(num_concepts, num_entries)).tocsr()
            concept_participation_in_postreqs = sparse.coo_matrix(
                (entries, (postreq_idxes, entry_idxes)),
                shape=(num_concepts, num_entries)).tocsr()
            concept_participation_in_prereq_edges = (
                concept_participation_in_prereqs,
                concept_participation_in_postreqs)
        else:
            prereq_edge_concept_idxes = concept_participation_in_prereq_edges = None

        last_student_embedding_idx = param_sizes[models.STUDENT_EMBEDDINGS]
        last_assessment_embedding_idx = last_student_embedding_idx + (
            param_sizes[models.ASSESSMENT_EMBEDDINGS])
        last_lesson_embedding_idx = last_assessment_embedding_idx + (
            param_sizes[models.LESSON_EMBEDDINGS]) if model.using_lessons else None
        last_prereq_embedding_idx = last_lesson_embedding_idx + (
            param_sizes[models.PREREQ_EMBEDDINGS]) if model.using_prereqs else None
        if model.using_lessons:
            last_student_bias_idx = (
                last_prereq_embedding_idx if model.using_prereqs else (
                    last_lesson_embedding_idx)) + param_sizes[models.STUDENT_BIASES]
        else:
            last_student_bias_idx = last_assessment_embedding_idx + (
                param_sizes[models.STUDENT_BIASES])

        last_assessment_bias_idx = last_student_bias_idx + param_sizes[models.ASSESSMENT_BIASES]
        (
            regularization_constant_for_student_embeddings,
            regularization_constant_for_assessment_embeddings,
            regularization_constant_for_lesson_embeddings,
            regularization_constant_for_prereq_embeddings,
            regularization_constant_for_concept_embeddings) = self.regularization_constant

        regularization_constants = np.zeros(len(param_vals))
        regularization_constants[:last_student_embedding_idx] = regularization_constant_for_student_embeddings
        regularization_constants[last_student_embedding_idx:last_assessment_embedding_idx] = regularization_constant_for_assessment_embeddings
        if model.using_lessons:
            regularization_constants[last_assessment_embedding_idx:last_lesson_embedding_idx] = regularization_constant_for_lesson_embeddings
            if model.using_prereqs:
                regularization_constants[last_lesson_embedding_idx:last_prereq_embedding_idx] = regularization_constant_for_prereq_embeddings
        if model.using_graph_prior:
            regularization_constants[last_assessment_bias_idx:] = regularization_constant_for_concept_embeddings

        box_constraints_of_parameters = {
            models.STUDENT_EMBEDDINGS : (
                model.anti_singularity_lower_bounds[models.STUDENT_EMBEDDINGS],
                None),
            models.ASSESSMENT_EMBEDDINGS : (
                model.anti_singularity_lower_bounds[models.ASSESSMENT_EMBEDDINGS],
                None),
            models.LESSON_EMBEDDINGS : (
                model.anti_singularity_lower_bounds[models.LESSON_EMBEDDINGS],
                None),
            models.PREREQ_EMBEDDINGS : (
                model.anti_singularity_lower_bounds[models.PREREQ_EMBEDDINGS],
                None),
            models.STUDENT_BIASES : (None, None) if model.using_bias else (0, 0),
            models.ASSESSMENT_BIASES : (None, None) if model.using_bias else (0, 0),
            models.CONCEPT_EMBEDDINGS : (
                model.anti_singularity_lower_bounds[models.CONCEPT_EMBEDDINGS],
                None) if model.using_graph_prior else None
        }
        box_constraints = np.concatenate(
            [[box_constraints_of_parameters[k]] * v.size for (
                k, v) in params.iteritems()])

        gradient_holder = np.zeros(param_vals.shape)

        grad_args = [
            assessment_interactions,
            lesson_interactions,
            model.learning_update_variance(times_since_prev_ixn_for_lesson_ixns),
            model.forgetting_penalty_terms(times_since_prev_ixn_for_lesson_ixns),
            self.regularization_constant,
            model.graph_regularization_constant,
            student_participation_in_assessment_ixns,
            student_bias_participation_in_assessment_ixns,
            assessment_participation_in_assessment_ixns,
            curr_student_participation_in_lesson_ixns,
            prev_student_participation_in_lesson_ixns,
            lesson_participation_in_lesson_ixns,
            assessment_participation_in_concepts,
            lesson_participation_in_concepts,
            concept_participation_in_assessments,
            concept_participation_in_lessons,
            prereq_edge_concept_idxes,
            concept_participation_in_prereq_edges,
            regularization_constants,
            last_student_embedding_idx,
            last_assessment_embedding_idx,
            last_lesson_embedding_idx,
            last_prereq_embedding_idx,
            last_student_bias_idx,
            last_assessment_bias_idx,
            model.history.duration(),
            model.using_bias,
            model.using_graph_prior,
            gradient_holder]

        if not self.using_scipy:
            # do not pass the gradient holder
            grad_args = grad_args[:-1]

            params = gradient_descent(
                grads(*grad_args),
                params,
                param_constraint_funcs=param_constraint_funcs,
                **self.gradient_descent_kwargs)

            # we originally passed in student embeddings
            # with shape (num_students, num_timesteps, embedding_dimension),
            # so let's switch back to (num_students,
            # embedding_dimension, num_timesteps)
            params[models.STUDENT_EMBEDDINGS] = np.reshape(
                params[models.STUDENT_EMBEDDINGS],
                (model.student_embeddings.shape[0],
                    model.student_embeddings.shape[2],
                    model.student_embeddings.shape[1])).swapaxes(1, 2)
        else:
            if self.verify_gradient:
                self.fd_err = optimize.check_grad(
                    (lambda x, args: grads(x, *args)[0]),
                    (lambda x, args: grads(x, *args)[1]),
                    param_vals,
                    tuple([param_shapes] + grad_args)) / math.sqrt(param_vals.size)

                if self.debug_mode_on:
                    _logger.debug(
                        'Root-mean-squared error of (forward) finite difference \
                        vs. analytic gradient = %f' % (self.fd_err))

            map_estimates = optimize.minimize(
                grads,
                param_vals,
                args=tuple([param_shapes] + grad_args),
                method='L-BFGS-B',
                jac=True,
                bounds=box_constraints,
                options={
                    'disp': self.debug_mode_on,
                    'ftol' : self.ftol
                    })

            # reshape parameter estimates from flattened array
            # into tensors and matrices
            params[models.STUDENT_EMBEDDINGS] = np.reshape(
                map_estimates.x[:last_student_embedding_idx],
                (model.student_embeddings.shape[0],
                    model.student_embeddings.shape[2],
                    model.student_embeddings.shape[1])).swapaxes(1, 2)

            params[models.ASSESSMENT_EMBEDDINGS] = np.reshape(
                map_estimates.x[last_student_embedding_idx:(
                    last_assessment_embedding_idx)],
                param_shapes[models.ASSESSMENT_EMBEDDINGS])

            begin_idx = last_assessment_embedding_idx
            if model.using_lessons:
                begin_idx = last_lesson_embedding_idx
                params[models.LESSON_EMBEDDINGS] = np.reshape(
                    map_estimates.x[last_assessment_embedding_idx:(
                        last_lesson_embedding_idx)],
                    param_shapes[models.LESSON_EMBEDDINGS])

            if model.using_prereqs:
                begin_idx = last_prereq_embedding_idx
                params[models.PREREQ_EMBEDDINGS] = np.reshape(
                    map_estimates.x[last_lesson_embedding_idx:(
                        last_prereq_embedding_idx)],
                    param_shapes[models.PREREQ_EMBEDDINGS])

            if model.using_bias:
                params[models.STUDENT_BIASES] = np.reshape(
                    map_estimates.x[begin_idx:last_student_bias_idx],
                    param_shapes[models.STUDENT_BIASES])
                params[models.ASSESSMENT_BIASES] = np.reshape(
                    map_estimates.x[last_student_bias_idx:last_assessment_bias_idx],
                    param_shapes[models.ASSESSMENT_BIASES])

            if model.using_graph_prior:
                params[models.CONCEPT_EMBEDDINGS] = np.reshape(
                    map_estimates.x[last_assessment_bias_idx:],
                    param_shapes[models.CONCEPT_EMBEDDINGS])

        # manually pin student state
        # after last interaction
        # (lack of interactions =>
        # no drift likelihoods in objective function to pin students)
        for student_id, t in timestep_of_last_interaction.iteritems():
            student_idx = model.history.idx_of_student_id(student_id)
            params[models.STUDENT_EMBEDDINGS][student_idx, :, (
                t-1):] = params[models.STUDENT_EMBEDDINGS][student_idx, :, t-1][:, None]

        model.student_embeddings = params[models.STUDENT_EMBEDDINGS]
        model.assessment_embeddings = params[models.ASSESSMENT_EMBEDDINGS]
        if model.using_lessons:
            model.lesson_embeddings = params[models.LESSON_EMBEDDINGS]
        if model.using_prereqs:
            model.prereq_embeddings = params[models.PREREQ_EMBEDDINGS]
        if model.using_bias:
            model.student_biases = params[models.STUDENT_BIASES]
            model.assessment_biases = params[models.ASSESSMENT_BIASES]
        if model.using_graph_prior:
            model.concept_embeddings = params[models.CONCEPT_EMBEDDINGS]
