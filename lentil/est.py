"""
Module for estimating model parameters

@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

from abc import abstractmethod
from collections import OrderedDict
import copy
import logging
import math
import time

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize, sparse

from . import models
from . import grad


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
    max_iter=1000,
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

        For example, see :py:func:`grad.without_scipy_*`

    :param dict[str,np.ndarray] params: Parameters allowed to vary
    :param dict[str,function] param_constraint_funcs: Functions that enforce bounds on parameters
        An example function for enforcing nonnegativity:
            
            lambda x: np.maximum(0, x)

    :param bool using_adagrad: Whether or not to use Adagrad
    :param float ftol: Stopping condition
        When relative difference between consecutive cost function evaluations
        drops below ftol, then the iterative optimization has "converged"

    :param float rate: Fixed learning rate
    :param float eta: Adagrad base learning rate
    :param float eps: Adagrad small epsilon
    :param bool debug_mode_on: True => dump plots using matplotlib.pyplot.show
    :param int num_checkpoints: Number of times to print "Iteration #" during updates
    :param int max_iter: Maximum number of training steps
    :rtype: dict[str,np.ndarray]
    :return: Parameter values at which gradient descent "converges"
    """
    if eta <= 0:
        raise ValueError('eta must be postive not {}'.format(eta))
    if eps <= 0:
        raise ValueError('eps must be positive not {}'.format(eps))
    if num_checkpoints <= 1:
        raise ValueError('Must have at least two checkpoints not {}'.format(num_checkpoints))
    if max_iter <= 0:
        raise ValueError('Maximum number of iterations must be strictly positive')

    if using_adagrad:
        # historical gradient (for Adagrad)
        hg = {k: np.zeros_like(v) for k, v in params.iteritems()}

    checkpoint_iter_step = max(1, max_iter // num_checkpoints)
    is_checkpoint = lambda idx: idx % checkpoint_iter_step == 0

    if verify_gradient:
        # check accuracy of gradient function using finite differences
        # TODO: move this check into a test suite and automate checking for correctness
        # for now, set verify_gradient=True in a notebook (e.g., nb/toy_examples.ipynb)
        # and look at the scatterplots/histograms yourself
        g, cst = grads(params)

        epsilon = 1e-9 # epsilon upper bound
        num_samples = 1 # number of epsilon samples to draw
        for k, v in g.iteritems():
            if v is None:
                continue
            fd = [] # gradient components computed with finite difference method
            cf = [] # gradient components computed with closed form expression
            for i, c in enumerate(v.ravel()):
                for _ in xrange(num_samples):
                    nparams = copy.deepcopy(params)
                    delta = epsilon * c

                    # flattening is necessary here because we are iterating over v.ravel()
                    nparams[k] = nparams[k].ravel()
                    nparams[k][i] += delta
                    nparams[k] = np.reshape(nparams[k], params[k].shape)
                    _, ncst = grads(nparams)

                    nparams[k] = nparams[k].ravel()
                    nparams[k][i] -= 2 * delta
                    nparams[k] = np.reshape(nparams[k], params[k].shape)
                    _, pcst = grads(nparams)

                    fd.append((ncst - pcst) / (2 * delta))
                cf.extend([c] * num_samples)

            _, ax = plt.subplots()
            ax.set_title('Components of Cost Gradient w.r.t. %s' % k)
            ax.set_xlabel('From Finite Differences')
            ax.set_ylabel('From Closed Form Expression')
            ax.scatter(fd, cf)
            plt.show()

            try:
                diffs = [(x-y)/x for x, y in zip(fd, cf)]
                _, ax = plt.subplots()
                ax.set_title('Diffs of Cost Gradient w.r.t. %s' % k)
                ax.set_xlabel(
                    '(From Finite Differences - From Closed Form Expression) \
                    / From Finite Differences')
                ax.set_ylabel('Frequency (number of gradient components)')
                ax.hist([x for x in diffs if not (np.isinf(x) or np.isnan(x))], bins=20)
                plt.show()
            except IndexError:
                pass

    costs = []
    gradient_norms = {k: [] for k in params}

    start_time = time.time()

    rel_diff = 2 * ftol # arbitrary starting point (should be greater than ftol)
    for iter_idx in xrange(max_iter):
        g, cst = grads(params)

        # don't use "for k, v in g.iteritems()", because we may be computing gradients
        # for parameters that are not being used
        for k in params:
            v = g[k]
            if using_adagrad:
                params[k] -= eta / (eps + np.sqrt(hg[k])) * v
                hg[k] += np.square(v)
            else:
                params[k] -= rate * v

            # projected gradient
            params[k] = param_constraint_funcs[k](params[k])

        for k in params:
            gradient_norms[k].append(np.linalg.norm(g[k]))

        if iter_idx >= 1:
            rel_diff = (cst - costs[-1]) / costs[-1]

        if is_checkpoint(iter_idx):
            _logger.debug('Iteration %d, rel_diff=%f', iter_idx, rel_diff)
            _logger.debug('Running at %f seconds per iteration',
                (time.time() - start_time) / (iter_idx + 1))

        costs.append(cst)
        
        if abs(rel_diff) <= ftol:
            break

    if debug_mode_on:
        _, ax = plt.subplots()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.plot(costs)
        plt.show()
        
        _, ax = plt.subplots()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L2 norm of gradient')
        for k, v in gradient_norms.iteritems():
            ax.plot(v, label=k)
        ax.legend(loc='upper right')
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
        max_iter=1000,
        gradient_descent_kwargs={},
        initial_param_vals={},
        using_scipy=True,
        verify_gradient=False,
        debug_mode_on=False,
        filtered_history=None,
        split_history=None):
        """
        Initialize estimator object

        :param float regularization_constant: Coefficient of L2 regularizer in cost function
        :param float ftol: Stopping condition
            When relative difference between consecutive cost function evaluations
            drops below ftol, then the iterative optimization has "converged"

        :param int max_iter: Maximum number of iterations for L-BFGS-B

        :param dict[str,object] gradient_descent_kwargs: Arguments for gradient_descent
        :param dict[str,np.ndarray] initial_param_vals: For warm starts
        :param bool using_scipy:
            True => use scipy.optimize.minimize,
            False => use batch gradient descent

        :param bool verify_gradient:
            True => use :py:func:`scipy.optimize.check_grad` to 
            verify accuracy of analytic gradient

        :param bool debug_mode_on:
            True => print checkpoints and dump plots

        :param pd.DataFrame filtered_history: A filtered history to be used instead of
            the history attached to the model passed to 
            :py:func:`est.EmbeddingMAPEstimator.fit_model`

            For details see :py:func:`datatools.InteractionHistory.split_interactions_by_type`

        :param datatools.SplitHistory split_history: An interaction history split into assessment
            interactions, lesson interactions, and timestep of last interaction for each student
        """
        if regularization_constant < 0:
            raise ValueError('regularization_constant must be nonnegative not {}'.format(
                regularization_constant))
        if ftol <= 0:
            raise ValueError('ftol must be positive not {}'.format(ftol))

        try:
            # if a number is passed, use same regularization constant for all embedding parameters
            # (students, assessments, lessons, prereqs, and concepts)
            regularization_constant = [float(regularization_constant)] * 5
        except TypeError:
            if len(regularization_constant) != 5:
                raise ValueError(
                        'regularization_constant must be either a number or a list of length 5')

        self.regularization_constant = regularization_constant
        self.gradient_descent_kwargs = gradient_descent_kwargs
        self.initial_param_vals = initial_param_vals
        self.filtered_history = filtered_history
        self.split_history = split_history
        self.ftol = ftol
        self.max_iter = max_iter

        self.using_scipy = using_scipy
        self.verify_gradient = verify_gradient
        self.debug_mode_on = debug_mode_on

    def fit_model(self, model):
        """
        Use iterative optimization to perform maximum a posteriori
        estimation of model parameters. Relies on hand-coded gradient.

        :param models.EmbeddingModel model: A skill embedding model
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

        constraint_func = lambda name: (lambda val: np.maximum(
            model.anti_singularity_lower_bounds[name], val))
        no_constraint_func = lambda x: x
        param_constraint_funcs = {
            models.STUDENT_EMBEDDINGS : constraint_func(models.STUDENT_EMBEDDINGS),
            models.ASSESSMENT_EMBEDDINGS : constraint_func(models.ASSESSMENT_EMBEDDINGS),
            models.LESSON_EMBEDDINGS : constraint_func(models.LESSON_EMBEDDINGS),
            models.PREREQ_EMBEDDINGS : constraint_func(models.PREREQ_EMBEDDINGS),
            models.STUDENT_BIASES : no_constraint_func,
            models.ASSESSMENT_BIASES : no_constraint_func,
            models.CONCEPT_EMBEDDINGS : constraint_func(models.CONCEPT_EMBEDDINGS),
        }

        params = OrderedDict()
        for key, value in param_shapes.iteritems():
            if key in self.initial_param_vals:
                params[key] = param_constraint_funcs[key](self.initial_param_vals[key])
            else:
                params[key] = param_constraint_funcs[key](np.random.random(value))

        if self.split_history is None:
            split_history = model.history.split_interactions_by_type(
                    filtered_history=self.filtered_history,
                    insert_dummy_lesson_ixns=True)
        else:
            split_history = self.split_history
        assessment_interactions = split_history.assessment_interactions
        lesson_interactions = split_history.lesson_interactions
        timestep_of_last_interaction = split_history.timestep_of_last_interaction

        if len(assessment_interactions) == 0:
            raise ValueError('No assessment interactions in history!')

        grads = grad.get_grad(
            using_scipy=self.using_scipy,
            using_lessons=model.using_lessons,
            using_prereqs=model.using_prereqs)

        #param_sizes = {k: v.size for k, v in params.iteritems()}
        param_sizes = {k: np.prod(v.shape) for k, v in params.iteritems()}
        param_vals = np.concatenate([v.ravel() for v in params.itervalues()], axis=0)

        (
            student_idxes_for_assessment_ixns,
            assessment_idxes_for_assessment_ixns, _) = assessment_interactions
            
        (
            student_idxes_for_lesson_ixns,
            lesson_idxes_for_lesson_ixns,
            times_since_prev_ixn_for_lesson_ixns) = lesson_interactions
        num_lesson_ixns = len(student_idxes_for_lesson_ixns)
        lesson_ixns_participation_matrix_entries = np.ones(num_lesson_ixns)
        
        num_assessment_ixns = len(student_idxes_for_assessment_ixns)
        assessment_ixns_participation_matrix_entries = np.ones(num_assessment_ixns)
        assessment_ixn_idxes = np.arange(num_assessment_ixns)

        # num_students * num_timesteps
        num_students_by_timesteps = param_shapes[models.STUDENT_EMBEDDINGS][0]
        
        num_assessments = param_shapes[models.ASSESSMENT_EMBEDDINGS][0]
        assessment_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (assessment_idxes_for_assessment_ixns, assessment_ixn_idxes)),
            shape=(num_assessments, num_assessment_ixns)).tocsr()
        student_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (student_idxes_for_assessment_ixns, assessment_ixn_idxes)),
            shape=(num_students_by_timesteps, num_assessment_ixns)).tocsr()

        num_timesteps = model.history.duration()
        student_bias_participation_in_assessment_ixns = sparse.coo_matrix(
            (assessment_ixns_participation_matrix_entries,
                (student_idxes_for_assessment_ixns // num_timesteps,
                    assessment_ixn_idxes)),
            shape=(num_students_by_timesteps // num_timesteps, num_assessment_ixns)).tocsr()

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

        # outside the "if model.using_lessons" statement
        # because we may need these for dummy lesson interactions in grad.*_without_lessons
        lesson_ixn_idxes = np.arange(num_lesson_ixns)
        curr_student_participation_in_lesson_ixns = sparse.coo_matrix(
            (lesson_ixns_participation_matrix_entries,
                (student_idxes_for_lesson_ixns, lesson_ixn_idxes)),
            shape=(num_students_by_timesteps, num_lesson_ixns)).tocsr()
        prev_student_participation_in_lesson_ixns = sparse.coo_matrix(
            (lesson_ixns_participation_matrix_entries,
                (student_idxes_for_lesson_ixns - 1, lesson_ixn_idxes)),
            shape=(num_students_by_timesteps, num_lesson_ixns)).tocsr()

        if not model.using_prereqs:
            # when computing gradients, we will be computing Ax - Bx
            # where A = curr_student_participation_in_lesson_ixns
            # and B = prev_student_participation_in_lesson_ixns,
            # so to speed things up we precompute A-B and compute (A-B)x
            curr_student_participation_in_lesson_ixns -= prev_student_participation_in_lesson_ixns

        if model.using_lessons:
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
                    (1 / num_concepts_per_lesson, (lesson_idxes, concept_idxes)),
                    shape=(num_lessons, num_concepts))
                concept_participation_in_lessons = lesson_participation_in_concepts.T
        else:
            lesson_participation_in_lesson_ixns = None
            lesson_participation_in_concepts = None
            concept_participation_in_lessons = None

        if model.using_graph_prior:
            prereq_idxes, postreq_idxes, num_concepts = model.graph.concept_prereq_edges()
            entries = np.ones(len(prereq_idxes))
            num_entries = len(entries)
            entry_idxes = np.arange(0, num_entries, 1)
            prereq_edge_concept_idxes = (prereq_idxes, postreq_idxes)
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
            if model.using_prereqs:
                last_student_bias_idx = last_prereq_embedding_idx
            else:
                last_student_bias_idx = last_lesson_embedding_idx
            last_student_bias_idx += param_sizes[models.STUDENT_BIASES]
        else:
            last_student_bias_idx = last_assessment_embedding_idx + \
                    param_sizes[models.STUDENT_BIASES]

        last_assessment_bias_idx = last_student_bias_idx + param_sizes[models.ASSESSMENT_BIASES]

        # dict[str,tuple(float|None,float|None)]
        # parameter name -> (lower bound, upper bound)
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
            [[box_constraints_of_parameters[k]] * v.size for (k, v) in params.iteritems()])

        gradient_holder = np.zeros(param_vals.shape)

        # TODO: pass these as kwargs
        # right now, we have to be very careful about the order of these arguments
        # and how they are received by functions in the grad module
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
            last_student_embedding_idx,
            last_assessment_embedding_idx,
            last_lesson_embedding_idx,
            last_prereq_embedding_idx,
            last_student_bias_idx,
            last_assessment_bias_idx,
            model.history.duration(),
            model.using_bias,
            model.using_graph_prior,
            model.using_l1_regularizer,
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
            # with shape (num_students * num_timesteps, embedding_dimension),
            # so let's switch back to (num_students, embedding_dimension, num_timesteps)
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

                _logger.debug(
                    'RMSE of (forward) finite difference vs. analytic gradient = %f', self.fd_err)

            map_estimates = optimize.minimize(
                grads,
                param_vals,
                args=tuple([param_shapes] + grad_args),
                method='L-BFGS-B',
                jac=True,
                bounds=box_constraints,
                options={
                    'disp': self.debug_mode_on,
                    'ftol' : self.ftol,
                    'maxiter' : self.max_iter
                    })

            # reshape parameter estimates from flattened array into tensors and matrices

            # we originally passed in student embeddings
            # with shape (num_students * num_timesteps, embedding_dimension),
            # so let's switch back to (num_students, embedding_dimension, num_timesteps)
            params[models.STUDENT_EMBEDDINGS] = np.reshape(
                map_estimates.x[:last_student_embedding_idx],
                (model.student_embeddings.shape[0],
                    model.student_embeddings.shape[2],
                    model.student_embeddings.shape[1])).swapaxes(1, 2)

            params[models.ASSESSMENT_EMBEDDINGS] = np.reshape(
                map_estimates.x[last_student_embedding_idx:last_assessment_embedding_idx],
                param_shapes[models.ASSESSMENT_EMBEDDINGS])

            begin_idx = last_assessment_embedding_idx
            if model.using_lessons:
                begin_idx = last_lesson_embedding_idx
                params[models.LESSON_EMBEDDINGS] = np.reshape(
                    map_estimates.x[last_assessment_embedding_idx:last_lesson_embedding_idx],
                    param_shapes[models.LESSON_EMBEDDINGS])

            if model.using_prereqs:
                begin_idx = last_prereq_embedding_idx
                params[models.PREREQ_EMBEDDINGS] = np.reshape(
                    map_estimates.x[last_lesson_embedding_idx:last_prereq_embedding_idx],
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

        # manually pin student state after last interaction 
        # lack of interactions => no drift likelihoods in objective function to pin students
        for student_id, t in timestep_of_last_interaction.iteritems():
            student_idx = model.history.idx_of_student_id(student_id)
            params[models.STUDENT_EMBEDDINGS][student_idx, :, t:] = \
                    params[models.STUDENT_EMBEDDINGS][student_idx, :, t][:, None]

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


class MIRTMAPEstimator(object):
    """
    Class for estimating parameters of multi-dimensional item response theory (MIRT) model
    by maximizing the log-likelihood of interactions (with L2 regularization on student 
    and assessment factors)
    """

    def __init__(
        self,
        regularization_constant=1e-3,
        ftol=1e-3,
        max_iter=1000,
        verify_gradient=False,
        debug_mode_on=False,
        filtered_history=None,
        split_history=None):
        """
        Initialize estimator object

        :param float regularization_constant: Coefficient of L2 regularizer in cost function
        :param float ftol: Stopping condition
            When relative difference between consecutive cost function evaluations
            drops below ftol, then the iterative optimization has "converged"

        :param int max_iter: Maximum number of iterations of BFGS
        :param bool verify_gradient: 
            True => use :py:func:`scipy.optimize.check_grad` to verify
            accuracy of analytic gradient

        :param bool debug_mode_on: True => display BFGS iterations
    
        :param pd.DataFrame|None filtered_history: A filtered history to be used instead of
            the history attached to the model passed to :py:func:`est.MIRTMAPEstimator.fit_model`

            For details see :py:func:`datatools.InteractionHistory.split_interactions_by_type`

        :param datatools.SplitHistory|None split_history: An interaction history split into
            assessment interactions, lesson interactions, and timestep of last interaction
            for each student
        """
        if regularization_constant < 0:
            raise ValueError('regularization_constant must be nonnegative not {}'.format(
                regularization_constant))
   
        if ftol <= 0:
            raise ValueError('ftol must be positive not {}'.format(ftol))

        self.regularization_constant = regularization_constant
        self.ftol = ftol
        self.max_iter = max_iter
        self.filtered_history = filtered_history
        self.split_history = split_history
        self.verify_gradient = verify_gradient
        self.debug_mode_on = debug_mode_on

    def fit_model(self, model):
        """
        Use iterative optimization to perform maximum a posteriori
        estimation of model parameters. Relies on hand-coded gradient.

        :param models.MIRTModel model: A mult-dimensional IRT model that needs to be fit
            to its interaction history
        """

        param_shapes = {
            models.STUDENT_FACTORS : model.student_factors.shape,
            models.ASSESSMENT_FACTORS : model.assessment_factors.shape
            }

        last_student_factor_idx = model.student_factors.size
        last_assessment_factor_idx = last_student_factor_idx + model.assessment_factors.size

        if self.split_history is None:
            split_history = model.history.split_interactions_by_type(
                    filtered_history=self.filtered_history,
                    insert_dummy_lesson_ixns=False)
        else:
            split_history = self.split_history
        assessment_interactions = split_history.assessment_interactions

        student_idxes_of_ixns, assessment_idxes_of_ixns, outcomes = assessment_interactions
        # TODO: explain
        student_idxes_of_ixns = student_idxes_of_ixns // model.history.duration()
        assessment_interactions = student_idxes_of_ixns, assessment_idxes_of_ixns, outcomes
        
        num_ixns = len(student_idxes_of_ixns)
        num_students = model.history.num_students()
        student_participation = sparse.coo_matrix(
            (np.ones(num_ixns), 
                (student_idxes_of_ixns, np.arange(0, num_ixns, 1))), 
            shape=(num_students, num_ixns)).tocsr()
        num_assessments = model.history.num_assessments()
        assessment_participation = sparse.coo_matrix(
            (np.ones(num_ixns),
                (assessment_idxes_of_ixns, np.arange(0, num_ixns, 1))),
            shape=(num_assessments, num_ixns)).tocsr()

        def gradient(
            params, 
            param_shapes,
            last_student_factor_idx,
            last_assessment_factor_idx,
            assessment_interactions, 
            student_participation, 
            assessment_participation,
            regularization_constant):
            """
            Evaluate cost function and its gradient at supplied parameter values

            :param np.array params: A flattened, concatenated array of parameter values
            :param dict[str,tuple] param_shapes: A dictionary mapping parameter name 
                to shape of np.ndarray

            :param int last_student_factor_idx: Index of last student factor parameter in 
                flattened params and flattened gradient

            :param int last_assessment_factor_idx: Index of last assessment factor parameter in
                flattened params and flattened gradient

            :param (np.array,np.array,np.array) assessment_interactions:
                A tuple of (student indices, assessment indices, outcomes) 
                for assessment interactions

            :param sparse.csr_array student_participation: A sparse binary matrix of shape
                [num_unique_students] X [num_assessment_interactions] that encodes which student
                was involved in each assessment interaction

            :param sparse.csr_array assessment_participation: A sparse binary matrix of shape
                [num_unique_assessments] X [num_assessment_interactions] that encodes which module
                was involved in each assessment interaction

            :param float regularization_constant: Coefficient of L2 regularization term
                for factors

            :rtype: (float,np.array)
            :return: A tuple of (cost, flattened and concatenated gradient)
            """

            g = np.empty(params.shape)

            student_factors = np.reshape(
                    params[:last_student_factor_idx],
                    param_shapes[models.STUDENT_FACTORS])

            assessment_factors = np.reshape(
                    params[last_student_factor_idx:last_assessment_factor_idx],
                    param_shapes[models.ASSESSMENT_FACTORS])

            assessment_offsets = params[last_assessment_factor_idx:]

            student_idxes_of_ixns, assessment_idxes_of_ixns, outcomes = assessment_interactions
            outcomes = outcomes[:, None]

            student_factors_of_ixns = student_factors[student_idxes_of_ixns, :]
            assessment_factors_of_ixns = assessment_factors[assessment_idxes_of_ixns, :]
            assessment_offsets_of_ixns = assessment_offsets[assessment_idxes_of_ixns][:, None]

            exp_diff = np.exp(-outcomes*(np.einsum(
                'ij, ij->i', 
                student_factors_of_ixns, 
                assessment_factors_of_ixns)[:, None] + assessment_offsets_of_ixns))
            one_plus_exp_diff = 1 + exp_diff
            mult_diff = outcomes * exp_diff / one_plus_exp_diff

            # gradient wrt student factors
            g[:last_student_factor_idx] = -student_participation.dot(
                    mult_diff * assessment_factors_of_ixns).ravel()

            # gradient wrt assessment factors
            g[last_student_factor_idx:last_assessment_factor_idx] = -assessment_participation.dot(
                    mult_diff * student_factors_of_ixns).ravel()

            # gradient from norm regularization
            g[:last_assessment_factor_idx] += \
                    2 * regularization_constant * params[:last_assessment_factor_idx]

            # gradient wrt assessment offsets
            g[last_assessment_factor_idx:] = -assessment_participation.dot(mult_diff)[:, 0]

            cost_from_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
            cost_from_norm_regularization = regularization_constant * (
                    params[:last_assessment_factor_idx]**2).sum()
            cost = cost_from_ixns + cost_from_norm_regularization

            return cost, g

        # random initialization
        params = np.concatenate((
            np.random.random(model.student_factors.shape).ravel(), 
            np.random.random(model.assessment_factors.shape).ravel(),
            np.random.random(model.assessment_offsets.shape)), axis=0)

        grad_args = (
                param_shapes,
                last_student_factor_idx,
                last_assessment_factor_idx,
                assessment_interactions,
                student_participation,
                assessment_participation,
                self.regularization_constant)
 
        if self.verify_gradient:
            self.fd_err = optimize.check_grad(
                (lambda x, args: gradient(x, *args)[0]),
                (lambda x, args: gradient(x, *args)[1]),
                params, grad_args) / math.sqrt(params.size)

            _logger.debug(
                'RMSE of (forward) finite difference vs. analytic gradient = %f', self.fd_err)

        bounds_on_assessment_factors = (None, None) if model.using_assessment_factors else (1, 1)
        bounds = [(None, None)] * model.student_factors.size + [(
            bounds_on_assessment_factors)] * model.assessment_factors.size + [(
                None, None)] * model.assessment_offsets.size

        map_estimates = optimize.minimize(
                gradient, params, args=grad_args, 
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'disp' : self.debug_mode_on, 
                    'ftol' : self.ftol, 
                    'maxiter' : self.max_iter})
        
        model.student_factors = np.reshape(
                map_estimates.x[:last_student_factor_idx],
                param_shapes[models.STUDENT_FACTORS])
        model.assessment_factors = np.reshape(
                map_estimates.x[last_student_factor_idx:last_assessment_factor_idx],
                param_shapes[models.ASSESSMENT_FACTORS])
        model.assessment_offsets = map_estimates.x[last_assessment_factor_idx:]

