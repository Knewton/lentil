"""
Module for gradients of the cost function in parameter estimation

@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import logging

import numpy as np

from . import models


_logger = logging.getLogger(__name__)


def without_scipy_without_lessons(
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer):
    """
    Setup a function that will compute gradients and evaluate the cost function
    at supplied parameter values, for an embedding model without lessons and
    a parameter estimation routine that uses gradient descent for optimization

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X
        [number of assessment interactions] where a non-zero entry indicates that the student
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X
        [number of assessment interactions] where a non-zero entry indicates that the assessment
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L2 regularization on lesson and assessment embeddings

    :rtype: function
    :return:
        A function that computes gradients and evaluates the cost function
        at supplied parameter values. See the docstring below for my_grads
        for further details.
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant

    def my_grads(param_vals):
        """
        Compute the gradient of the cost function with respect to model parameters

        :param dict[str,np.ndarray] param_vals:
            A dictionary mapping a parameter's name to its current value

        :rtype: (dict[str,np.ndarray], float)
        :return:
            A dictionary mapping a parameter's name to the gradient
            of the cost function with respect to that parameter
            (evaluated at the supplied parameter values), and the value of the cost function
            (evaluated at the supplied parameter values)
        """

        # pull parameters from param_vals into separate variables
        student_embeddings = param_vals[models.STUDENT_EMBEDDINGS]
        assessment_embeddings = param_vals[models.ASSESSMENT_EMBEDDINGS]
        if using_graph_prior:
            concept_embeddings = param_vals[models.CONCEPT_EMBEDDINGS]

        # split assessment interactions into students, assessments, outcomes
        (
            student_idxes_for_assessment_ixns,
            assessment_idxes_for_assessment_ixns,
            outcomes_for_assessment_ixns) = assessment_interactions

        # use dummy lesson interactions to get students in temporal process
        student_idxes_for_temporal_process, _, _ = lesson_interactions

        # get biases for assessment interactions
        if using_bias:
            student_biases = param_vals[models.STUDENT_BIASES][\
                    student_idxes_for_assessment_ixns // num_timesteps][:, None]
            assessment_biases = param_vals[models.ASSESSMENT_BIASES][\
                    assessment_idxes_for_assessment_ixns][:, None]
        else:
            student_biases = assessment_biases = 0

        # shape outcomes as a column vector
        outcomes = outcomes_for_assessment_ixns[:, None]

        # get the assessment embedding for each assessment interaction
        assessment_embeddings_for_assessment_ixns = \
                assessment_embeddings[assessment_idxes_for_assessment_ixns, :]

        # compute the L2 norm of the assessment embedding for each assessment interaction
        assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
            assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

        # get the student embedding for each assessment interaction
        student_embeddings_for_assessment_ixns = \
                student_embeddings[student_idxes_for_assessment_ixns, :]

        # compute the dot product of the student embedding
        # and assessment embedding for each interaction
        student_dot_assessment = np.einsum(
            'ij, ij->i',
            student_embeddings_for_assessment_ixns,
            assessment_embeddings_for_assessment_ixns)[:, None]

        # compute intermediate quantities for the gradient that get reused
        exp_diff = np.exp(outcomes * (
            assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                    assessment_embedding_norms_for_assessment_ixns - student_biases - \
                    assessment_biases))
        one_plus_exp_diff = 1 + exp_diff
        mult_diff = outcomes * exp_diff / one_plus_exp_diff

        using_temporal_process = len(student_idxes_for_temporal_process) > 0
        if using_temporal_process:
            # get embeddings of student states resulting from lesson interactions
            curr_student_embeddings_for_lesson_ixns = \
                    student_embeddings[student_idxes_for_temporal_process, :]

            # get embeddings of student states prior to lesson interactions
            prev_student_embeddings_for_lesson_ixns = \
                    student_embeddings[student_idxes_for_temporal_process - 1, :]

            # compute intermediate quantities for the gradient that get reused
            diffs = curr_student_embeddings_for_lesson_ixns - \
                    prev_student_embeddings_for_lesson_ixns + forgetting_penalty_terms
            diffs_over_var = diffs / learning_update_variance
        else:
            diffs = diffs_over_var = 0

         # compute intermediate quantities for graph regularization terms in the gradient
        if using_graph_prior:
            # get distance from an assessment embedding to its prior embedding,
            # i.e., the weighted average of the embeddings of the assessment's
            # governing concepts
            assessment_diffs_from_concept_centers = assessment_embeddings - \
                    assessment_participation_in_concepts.dot(concept_embeddings)

            # grab the concept dependency graph
            prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
            concept_participation_in_prereqs, concept_participation_in_postreqs = \
                    concept_participation_in_prereq_edges

            # get prereq and postreq concept embeddings
            prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
            postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

            # compute column vector of L2 norms for postreq concept embeddings
            postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]

            # compute the dot product of the prereq concept embedding
            # and postreq concept embedding for each edge in the concept dependency graph
            prereq_dot_postreq = np.einsum(
                'ij, ij->i',
                prereq_concept_embeddings,
                postreq_concept_embeddings)[:, None]

            # intermediate quantity, useful and reusable later
            prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

        # compute the gradient w.r.t. student embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
            mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                    assessment_embeddings_for_assessment_ixns)
        stud_grad_from_norm_regularization = 2 * student_regularization_constant * \
                student_embeddings
        if using_temporal_process:
            stud_grad_from_temporal_process = curr_student_participation_in_lesson_ixns.dot(
                    diffs_over_var)
        else:
            stud_grad_from_temporal_process = 0
        gradient_wrt_student_embedding = stud_grad_from_asmt_ixns + \
                stud_grad_from_norm_regularization + + stud_grad_from_temporal_process

        # compute the gradient w.r.t. assessment embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
                mult_diff / assessment_embedding_norms_for_assessment_ixns * (
                        student_embeddings_for_assessment_ixns - \
                                assessment_embeddings_for_assessment_ixns - \
                                student_dot_assessment / np.einsum(
                                    'ij, ij->ij',
                                    assessment_embedding_norms_for_assessment_ixns,
                                    assessment_embedding_norms_for_assessment_ixns) * \
                                            assessment_embeddings_for_assessment_ixns))
        if using_l1_regularizer:
            asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                    assessment_embeddings)
        else:
            asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                assessment_embeddings
        if using_graph_prior:
            asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                    assessment_diffs_from_concept_centers
        else:
            asmt_grad_from_graph_regularization = 0
        gradient_wrt_assessment_embedding = asmt_grad_from_asmt_ixns + \
                asmt_grad_from_norm_regularization + asmt_grad_from_graph_regularization

        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_student_biases = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_assessment_biases = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        if using_graph_prior:
            # compute the gradient w.r.t. concept embeddings,
            # which is the sum of gradient of the log-likelihood
            # of assessment and lesson interactions and the gradient
            # of the regularization terms
            concept_grad_from_assessments = -concept_participation_in_assessments.dot(
                2 * assessment_diffs_from_concept_centers)

            concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
                postreq_concept_embeddings / postreq_concept_norms)

            concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
                (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / \
                        postreq_concept_norms + 2 * postreq_concept_embeddings * \
                        prereq_dot_postreq / postreq_concept_norms**3)

            gradient_wrt_concept_embedding = graph_regularization_constant * (
                concept_grad_from_assessments + concept_grad_from_prereqs + \
                        concept_grad_from_postreqs) + 2 * concept_regularization_constant * \
                        concept_embeddings
        else:
            gradient_wrt_concept_embedding = None

        gradient = {
            models.STUDENT_EMBEDDINGS : gradient_wrt_student_embedding,
            models.ASSESSMENT_EMBEDDINGS : gradient_wrt_assessment_embedding,
            models.STUDENT_BIASES : gradient_wrt_student_biases,
            models.ASSESSMENT_BIASES : gradient_wrt_assessment_biases,
            models.CONCEPT_EMBEDDINGS : gradient_wrt_concept_embedding
            }

        cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
        if using_temporal_process:
            cost_from_temporal_process = np.einsum(
                'ij, ij', diffs, diffs) / (2 * learning_update_variance)
        else:
            cost_from_temporal_process = 0
        cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
        if using_l1_regularizer:
            cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                    assessment_embeddings).sum()
        else:
            cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)

        if using_graph_prior:
            cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
            cost_from_graph_regularization = graph_regularization_constant * (
                (assessment_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
        else:
            cost_from_concept_regularization = cost_from_graph_regularization = 0

        cost_from_regularization = cost_from_student_regularization + \
                cost_from_assessment_regularization + cost_from_concept_regularization + \
                cost_from_graph_regularization
        cost = cost_from_assessment_ixns + cost_from_temporal_process + cost_from_regularization

        return gradient, cost

    return my_grads

def without_scipy_with_prereqs(
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer):
    """
    Setup a function that will compute gradients and evaluate the cost function
    at supplied parameter values, for a full embedding model and
    a parameter estimation routine that uses gradient descent for optimization

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X
        [number of assessment interactions] where a non-zero entry indicates that the student
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X
        [number of assessment interactions] where a non-zero entry indicates that the assessment
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L2 regularization on lesson and assessment embeddings

    :rtype: function
    :return:
        A function that computes gradients and evaluates the cost function
        at supplied parameter values. See the docstring below for my_grads
        for further details.
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant

    def my_grads(param_vals):
        """
        Compute the gradient of the cost function with respect to model parameters

        :param dict[str,np.ndarray] param_vals:
            A dictionary mapping a parameter's name to its current value

        :rtype: (dict[str,np.ndarray],float)
        :return:
            A dictionary mapping a parameter's name to the gradient
            of the cost function with respect to that parameter
            (evaluated at the supplied parameter values),
            and the value of the cost function
            (evaluated at the supplied parameter values)
        """

        # pull parameters from param_vals into separate variables
        student_embeddings = param_vals[models.STUDENT_EMBEDDINGS]
        assessment_embeddings = param_vals[models.ASSESSMENT_EMBEDDINGS]
        lesson_embeddings = param_vals[models.LESSON_EMBEDDINGS]
        prereq_embeddings = param_vals[models.PREREQ_EMBEDDINGS]
        if using_graph_prior:
            concept_embeddings = param_vals[models.CONCEPT_EMBEDDINGS]

        # split assessment interactions into students, assessments, outcomes
        (
            student_idxes_for_assessment_ixns,
            assessment_idxes_for_assessment_ixns,
            outcomes_for_assessment_ixns) = assessment_interactions

        # split lesson interactions into students, lessons
        student_idxes_for_lesson_ixns, lesson_idxes_for_lesson_ixns, _ = lesson_interactions

        # get biases for assessment interactions
        if using_bias:
            student_biases = param_vals[models.STUDENT_BIASES][\
                    student_idxes_for_assessment_ixns // num_timesteps][:, None]
            assessment_biases = param_vals[models.ASSESSMENT_BIASES][\
                    assessment_idxes_for_assessment_ixns][:, None]
        else:
            student_biases = assessment_biases = 0

        # shape outcomes as a column vector
        outcomes = outcomes_for_assessment_ixns[:, None]

        # get the assessment embedding for each assessment interaction
        assessment_embeddings_for_assessment_ixns = \
                assessment_embeddings[assessment_idxes_for_assessment_ixns, :]

        # compute the L2 norm of the assessment embedding for each assessment interaction
        assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
            assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

        # get the student embedding for each assessment interaction
        student_embeddings_for_assessment_ixns = \
                student_embeddings[student_idxes_for_assessment_ixns, :]

        # compute the dot product of the student embedding
        # and assessment embedding for each interaction
        student_dot_assessment = np.einsum(
            'ij, ij->i',
            student_embeddings_for_assessment_ixns,
            assessment_embeddings_for_assessment_ixns)[:, None]

        # compute intermediate quantities for the gradient that get reused
        exp_diff = np.exp(outcomes * (
            assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                    assessment_embedding_norms_for_assessment_ixns - student_biases - \
                    assessment_biases))
        one_plus_exp_diff = 1 + exp_diff
        mult_diff = outcomes * exp_diff / one_plus_exp_diff

        # get lesson embeddings for lesson interactions
        lesson_embeddings_for_lesson_ixns = lesson_embeddings[lesson_idxes_for_lesson_ixns, :]

        # get lesson prereq embeddings for lesson interactions
        prereq_embeddings_for_lesson_ixns = prereq_embeddings[lesson_idxes_for_lesson_ixns, :]

        # get embeddings of student states resulting from lesson interactions
        curr_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_lesson_ixns, :]

        # get embeddings of student states prior to lesson interactions
        prev_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_lesson_ixns - 1, :]

        # compute the L2 norm of the lesson embedding for each lesson interaction
        prereq_embedding_norms_for_lesson_ixns = np.linalg.norm(
            prereq_embeddings_for_lesson_ixns, axis=1)[:, None]

        # compute the dot product of the student embedding prior
        # to the lesson interaction and the lesson prereq embedding,
        # for each interaction
        prev_student_dot_prereq = np.einsum(
            'ij, ij->i',
            prev_student_embeddings_for_lesson_ixns,
            prereq_embeddings_for_lesson_ixns)[:, None]

        # compute intermediate quantities for the gradient that get reused
        update_exp_diff = np.exp(
            prereq_embedding_norms_for_lesson_ixns - prev_student_dot_prereq / \
                    prereq_embedding_norms_for_lesson_ixns)
        update_one_plus_exp_diff = 1 + update_exp_diff
        diffs = curr_student_embeddings_for_lesson_ixns - prev_student_embeddings_for_lesson_ixns \
                - lesson_embeddings_for_lesson_ixns / update_one_plus_exp_diff + \
                forgetting_penalty_terms
        diffs_over_var = diffs / learning_update_variance
        update_mult_diff = np.einsum(
            'ij, ij->i',
            diffs_over_var,
            lesson_embeddings_for_lesson_ixns)[:, None] * update_exp_diff / (
                    np.einsum(
                        'ij, ij->ij',
                        update_one_plus_exp_diff,
                        update_one_plus_exp_diff) * prereq_embedding_norms_for_lesson_ixns)

        if using_graph_prior:
            # get distance from an assessment embedding to its prior embedding,
            # i.e., the weighted average of the embeddings of the assessment's
            # governing concepts
            assessment_diffs_from_concept_centers = assessment_embeddings - \
                    assessment_participation_in_concepts.dot(concept_embeddings)

            # get distance from a lesson embedding to its prior embedding,
            # i.e., the weighted average of the embeddings of the lesson's
            # governing concepts
            lesson_diffs_from_concept_centers = lesson_embeddings - \
                    lesson_participation_in_concepts.dot(concept_embeddings)

            # grab the concept dependency graph
            prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
            concept_participation_in_prereqs, concept_participation_in_postreqs = \
                    concept_participation_in_prereq_edges

            # get prereq and postreq concept embeddings
            prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
            postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

            # compute column vector of L2 norms for postreq concept embeddings
            postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]

            # compute the dot product of the prereq concept embedding
            # and postreq concept embedding for each edge in the concept dependency graph
            prereq_dot_postreq = np.einsum(
                'ij, ij->i',
                prereq_concept_embeddings,
                postreq_concept_embeddings)[:, None]

            # intermediate quantity, useful and reusable later
            prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

        # compute the gradient w.r.t. student embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
                mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                        assessment_embeddings_for_assessment_ixns)
        stud_grad_from_lesson_ixns = curr_student_participation_in_lesson_ixns.dot(
            diffs_over_var) - prev_student_participation_in_lesson_ixns.dot(
            update_mult_diff * prereq_embeddings_for_lesson_ixns + diffs_over_var)
        stud_grad_from_norm_regularization = 2 * student_regularization_constant * \
                student_embeddings
        gradient_wrt_student_embedding = stud_grad_from_asmt_ixns + stud_grad_from_lesson_ixns + \
                stud_grad_from_norm_regularization

        # compute the gradient w.r.t. assessment embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
                mult_diff / assessment_embedding_norms_for_assessment_ixns * (
                        student_embeddings_for_assessment_ixns - \
                                assessment_embeddings_for_assessment_ixns - \
                                student_dot_assessment / np.einsum(
                                    'ij, ij->ij',
                                    assessment_embedding_norms_for_assessment_ixns,
                                    assessment_embedding_norms_for_assessment_ixns) * \
                                            assessment_embeddings_for_assessment_ixns))
        if using_l1_regularizer:
            asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                    assessment_embeddings)
        else:
            asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                    assessment_embeddings
        if using_graph_prior:
            asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                    assessment_diffs_from_concept_centers
        else:
            asmt_grad_from_graph_regularization = 0
        gradient_wrt_assessment_embedding = asmt_grad_from_asmt_ixns + \
                asmt_grad_from_norm_regularization + asmt_grad_from_graph_regularization

        # compute the gradient w.r.t. lesson embeddings,
        # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
        # and the gradient of the regularization terms
        lesson_grad_from_lesson_ixns = -lesson_participation_in_lesson_ixns.dot(
            diffs_over_var / update_one_plus_exp_diff)
        if using_l1_regularizer:
            lesson_grad_from_norm_regularization = lesson_regularization_constant * np.sign(
                    lesson_embeddings)
        else:
            lesson_grad_from_norm_regularization = 2 * lesson_regularization_constant * \
                    lesson_embeddings
        if using_graph_prior:
            lesson_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                    lesson_diffs_from_concept_centers
        else:
            lesson_grad_from_graph_regularization = 0
        gradient_wrt_lesson_embedding = lesson_grad_from_lesson_ixns + \
                lesson_grad_from_norm_regularization + lesson_grad_from_graph_regularization

        # compute the gradient w.r.t. prereq embeddings,
        # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
        # and the gradient of the regularization terms
        prereq_grad_from_lesson_ixns = lesson_participation_in_lesson_ixns.dot(
            update_mult_diff * (prev_student_dot_prereq / np.einsum(
                'ij, ij->ij',
                prereq_embedding_norms_for_lesson_ixns,
                prereq_embedding_norms_for_lesson_ixns) * \
                        prereq_embeddings_for_lesson_ixns - \
                        prev_student_embeddings_for_lesson_ixns + \
                        prereq_embeddings_for_lesson_ixns))
        prereq_grad_from_norm_regularization = 2 * prereq_regularization_constant * \
                prereq_embeddings
        gradient_wrt_prereq_embedding = \
                prereq_grad_from_lesson_ixns + prereq_grad_from_norm_regularization

        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_student_biases = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_assessment_biases = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        if using_graph_prior:
            # compute the gradient w.r.t. concept embeddings,
            # which is the sum of gradient of the log-likelihood of assessment
            # and lesson interactions and the gradient of the regularization terms
            concept_grad_from_assessments = -concept_participation_in_assessments.dot(
                2 * assessment_diffs_from_concept_centers)

            concept_grad_from_lessons = concept_participation_in_lessons.dot(
                2 * lesson_diffs_from_concept_centers)

            concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
                postreq_concept_embeddings / postreq_concept_norms)

            concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
                (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / \
                        postreq_concept_norms - 2 * prereq_dot_postreq * \
                        postreq_concept_embeddings / postreq_concept_norms**3)

            gradient_wrt_concept_embedding = graph_regularization_constant * (
                concept_grad_from_assessments + concept_grad_from_lessons + \
                        concept_grad_from_prereqs + concept_grad_from_postreqs)
        else:
            gradient_wrt_concept_embedding = None

        gradient = {
            models.STUDENT_EMBEDDINGS : gradient_wrt_student_embedding,
            models.ASSESSMENT_EMBEDDINGS : gradient_wrt_assessment_embedding,
            models.LESSON_EMBEDDINGS : gradient_wrt_lesson_embedding,
            models.PREREQ_EMBEDDINGS : gradient_wrt_prereq_embedding,
            models.STUDENT_BIASES : gradient_wrt_student_biases,
            models.ASSESSMENT_BIASES : gradient_wrt_assessment_biases,
            models.CONCEPT_EMBEDDINGS : gradient_wrt_concept_embedding
            }

        cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
        cost_from_lesson_ixns = np.einsum('ij, ij', diffs, diffs) / (2 * learning_update_variance)

        cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
        if using_l1_regularizer:
            cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                    assessment_embeddings).sum()
            cost_from_lesson_regularization = lesson_regularization_constant * np.absolute(
                    lesson_embeddings).sum()
        else:
            cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)
            cost_from_lesson_regularization = lesson_regularization_constant * np.einsum(
                'ij, ij', lesson_embeddings, lesson_embeddings)
        cost_from_prereq_regularization = prereq_regularization_constant * np.einsum(
            'ij, ij', prereq_embeddings, prereq_embeddings)

        if using_graph_prior:
            cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
            cost_from_graph_regularization = graph_regularization_constant * ((
                assessment_diffs_from_concept_centers**2).sum() + (
                lesson_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
        else:
            cost_from_concept_regularization = cost_from_graph_regularization = 0

        cost_from_ixns = cost_from_assessment_ixns + cost_from_lesson_ixns
        cost_from_regularization = cost_from_student_regularization + \
                cost_from_assessment_regularization + cost_from_lesson_regularization + \
                cost_from_prereq_regularization + cost_from_concept_regularization + \
                cost_from_graph_regularization
        cost = cost_from_ixns + cost_from_regularization

        return gradient, cost

    return my_grads

def without_scipy_without_prereqs(
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer):
    """
    Setup a function that will compute gradients and evaluate the cost function
    at supplied parameter values, for an embedding model without prereqs and
    a parameter estimation routine that uses gradient descent for optimization

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X
        [number of assessment interactions] where a non-zero entry indicates that the student
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X
        [number of assessment interactions] where a non-zero entry indicates that the assessment
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L2 regularization on lesson and assessment embeddings

    :rtype: function
    :return:
        A function that computes gradients and evaluates the cost function
        at supplied parameter values. See the docstring below for my_grads
        for further details.
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant

    def my_grads(param_vals):
        """
        Compute the gradient of the cost function with respect to model parameters

        :param dict[str,np.ndarray] param_vals:
            A dictionary mapping a parameter's name to its current value

        :rtype: (dict[str,np.ndarray],float)
        :return:
            A dictionary mapping a parameter's name to the gradient
            of the cost function with respect to that parameter
            (evaluated at the supplied parameter values),
            and the value of the cost function
            (evaluated at the supplied parameter values)
        """

        # pull parameters from param_vals into separate variables
        student_embeddings = param_vals[models.STUDENT_EMBEDDINGS]
        assessment_embeddings = param_vals[models.ASSESSMENT_EMBEDDINGS]
        lesson_embeddings = param_vals[models.LESSON_EMBEDDINGS]
        if using_graph_prior:
            concept_embeddings = param_vals[models.CONCEPT_EMBEDDINGS]

        # split assessment interactions into students, assessments, outcomes
        (
            student_idxes_for_assessment_ixns,
            assessment_idxes_for_assessment_ixns,
            outcomes_for_assessment_ixns) = assessment_interactions

        # split lesson interactions into students, lessons
        student_idxes_for_lesson_ixns, lesson_idxes_for_lesson_ixns, _ = lesson_interactions

        # get biases for assessment interactions
        if using_bias:
            student_biases = param_vals[models.STUDENT_BIASES][\
                    student_idxes_for_assessment_ixns // num_timesteps][:, None]
            assessment_biases = param_vals[models.ASSESSMENT_BIASES][\
                    assessment_idxes_for_assessment_ixns][:, None]
        else:
            student_biases = assessment_biases = 0

        # shape outcomes as a column vector
        outcomes = outcomes_for_assessment_ixns[:, None]

        # get the assessment embedding for each assessment interaction
        assessment_embeddings_for_assessment_ixns = \
                assessment_embeddings[assessment_idxes_for_assessment_ixns, :]

        # compute the L2 norm of the assessment embedding for each assessment interaction
        assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
            assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

        # get the student embedding for each assessment interaction
        student_embeddings_for_assessment_ixns = \
                student_embeddings[student_idxes_for_assessment_ixns, :]

        # compute the dot product of the student embedding
        # and assessment embedding for each interaction
        student_dot_assessment = np.einsum(
            'ij, ij->i',
            student_embeddings_for_assessment_ixns,
            assessment_embeddings_for_assessment_ixns)[:, None]

        # compute intermediate quantities for the gradient that get reused
        exp_diff = np.exp(outcomes * (
            assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                    assessment_embedding_norms_for_assessment_ixns - student_biases - \
                    assessment_biases))
        one_plus_exp_diff = 1 + exp_diff
        mult_diff = outcomes * exp_diff / one_plus_exp_diff

        # get lesson embeddings for lesson interactions
        lesson_embeddings_for_lesson_ixns = lesson_embeddings[lesson_idxes_for_lesson_ixns, :]

        # get embeddings of student states resulting from lesson interactions
        curr_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_lesson_ixns, :]

        # get embeddings of student states prior to lesson interactions
        prev_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_lesson_ixns - 1, :]

        # compute intermediate quantities for the gradient that get reused
        diffs = curr_student_embeddings_for_lesson_ixns - prev_student_embeddings_for_lesson_ixns \
                - lesson_embeddings_for_lesson_ixns + forgetting_penalty_terms
        diffs_over_var = diffs / learning_update_variance

        if using_graph_prior:
            # get distance from an assessment embedding to its prior embedding,
            # i.e., the weighted average of the embeddings of the assessment's
            # governing concepts
            assessment_diffs_from_concept_centers = assessment_embeddings - \
                    assessment_participation_in_concepts.dot(concept_embeddings)

            # get distance from a lesson embedding to its prior embedding,
            # i.e., the weighted average of the embeddings of the lesson's
            # governing concepts
            lesson_diffs_from_concept_centers = lesson_embeddings - \
                    lesson_participation_in_concepts.dot(concept_embeddings)

            # grab the concept dependency graph
            prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
            concept_participation_in_prereqs, concept_participation_in_postreqs = \
                    concept_participation_in_prereq_edges

            # get prereq and postreq concept embeddings
            prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
            postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

            # compute column vector of L2 norms for postreq concept embeddings
            postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]

            # compute the dot product of the prereq concept embedding
            # and postreq concept embedding for each edge in the concept dependency graph
            prereq_dot_postreq = np.einsum(
                'ij, ij->i',
                prereq_concept_embeddings,
                postreq_concept_embeddings)[:, None]

            # intermediate quantity, useful and reusable later
            prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

        # compute the gradient w.r.t. student embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
            mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                    assessment_embeddings_for_assessment_ixns)
        stud_grad_from_lesson_ixns = curr_student_participation_in_lesson_ixns.dot(diffs_over_var)
        stud_grad_from_norm_regularization = 2 * student_regularization_constant * \
                student_embeddings
        gradient_wrt_student_embedding = stud_grad_from_asmt_ixns + stud_grad_from_lesson_ixns + \
                stud_grad_from_norm_regularization

        # compute the gradient w.r.t. assessment embeddings,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        # and the gradient of the regularization terms
        asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
                mult_diff / assessment_embedding_norms_for_assessment_ixns * (
                    student_embeddings_for_assessment_ixns - \
                            assessment_embeddings_for_assessment_ixns - student_dot_assessment / \
                            np.einsum(
                                'ij, ij->ij',
                                assessment_embedding_norms_for_assessment_ixns,
                                assessment_embedding_norms_for_assessment_ixns) * \
                                        assessment_embeddings_for_assessment_ixns))
        if using_l1_regularizer:
            asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                    assessment_embeddings)
        else:
            asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                    assessment_embeddings
        if using_graph_prior:
            asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                    assessment_diffs_from_concept_centers
        else:
            asmt_grad_from_graph_regularization = 0
        gradient_wrt_assessment_embedding = asmt_grad_from_asmt_ixns + \
                asmt_grad_from_norm_regularization + asmt_grad_from_graph_regularization

        # compute the gradient w.r.t. lesson embeddings,
        # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
        # and the gradient of the regularization terms
        lesson_grad_from_lesson_ixns = -lesson_participation_in_lesson_ixns.dot(diffs_over_var)
        if using_l1_regularizer:
            lesson_grad_from_norm_regularization = lesson_regularization_constant * np.sign(
                    lesson_embeddings)
        else:
            lesson_grad_from_norm_regularization = 2 * lesson_regularization_constant * \
                    lesson_embeddings
        if using_graph_prior:
            lesson_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                    lesson_diffs_from_concept_centers
        else:
            lesson_grad_from_graph_regularization = 0
        gradient_wrt_lesson_embedding = lesson_grad_from_lesson_ixns + \
                lesson_grad_from_norm_regularization + lesson_grad_from_graph_regularization

        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_student_biases = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient_wrt_assessment_biases = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff)[:,0]

        if using_graph_prior:
            # compute the gradient w.r.t. concept embeddings,
            # which is the sum of gradient of the log-likelihood of assessment
            # and lesson interactions and the gradient of the regularization terms
            concept_grad_from_assessments = -concept_participation_in_assessments.dot(
                2 * assessment_diffs_from_concept_centers)

            concept_grad_from_lessons = concept_participation_in_lessons.dot(
                2 * lesson_diffs_from_concept_centers)

            concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
                postreq_concept_embeddings / postreq_concept_norms)

            concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
                (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / \
                        postreq_concept_norms - 2 * prereq_dot_postreq * \
                        postreq_concept_embeddings / postreq_concept_norms**3)

            gradient_wrt_concept_embedding = graph_regularization_constant * (
                concept_grad_from_assessments + concept_grad_from_lessons + \
                        concept_grad_from_prereqs + concept_grad_from_postreqs)
        else:
            gradient_wrt_concept_embedding = None

        gradient = {
            models.STUDENT_EMBEDDINGS : gradient_wrt_student_embedding,
            models.ASSESSMENT_EMBEDDINGS : gradient_wrt_assessment_embedding,
            models.LESSON_EMBEDDINGS : gradient_wrt_lesson_embedding,
            models.STUDENT_BIASES : gradient_wrt_student_biases,
            models.ASSESSMENT_BIASES : gradient_wrt_assessment_biases,
            models.CONCEPT_EMBEDDINGS : gradient_wrt_concept_embedding
            }

        cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
        cost_from_lesson_ixns = np.einsum('ij, ij', diffs, diffs) / (2 * learning_update_variance)

        cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
        if using_l1_regularizer:
            cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                    assessment_embeddings).sum()
            cost_from_lesson_regularization = lesson_regularization_constant * np.absolute(
                    lesson_embeddings).sum()
        else:
            cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)
            cost_from_lesson_regularization = lesson_regularization_constant * np.einsum(
                'ij, ij', lesson_embeddings, lesson_embeddings)

        if using_graph_prior:
            cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
            cost_from_graph_regularization = graph_regularization_constant * ((
                assessment_diffs_from_concept_centers**2).sum() + (
                lesson_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
        else:
            cost_from_concept_regularization = cost_from_graph_regularization = 0

        cost_from_ixns = cost_from_assessment_ixns + cost_from_lesson_ixns
        cost_from_regularization = cost_from_student_regularization + \
                cost_from_assessment_regularization + cost_from_lesson_regularization + \
                cost_from_concept_regularization + cost_from_graph_regularization
        cost = cost_from_ixns + cost_from_regularization

        return gradient, cost

    return my_grads

def with_scipy_without_lessons(
    param_vals,
    param_shapes,
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer,
    gradient):
    """
    Compute the gradient of the cost function with respect to model parameters

    :param np.array param_vals:
        Flattened, concatenated parameter values

    :param dict[str,tuple] param_shapes:
        A dictionary mapping a parameter's name to the shape of its np.ndarray

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X
        [number of assessment interactions] where a non-zero entry indicates that the student
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X
        [number of assessment interactions] where a non-zero entry indicates that the assessment
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L1 regularization on lesson and assessment embeddings

    :param np.array gradient:
        Placeholder for the flattened gradient

    :rtype: (float,np.array)
    :return:
        The value of the cost function
        (evaluated at the supplied parameter values),
        and the flattened gradient of the cost function
        (evaluated at the supplied parameter values)
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant
    
    # reshape flattened student embeddings into tensor
    student_embeddings = np.reshape(
        param_vals[:last_student_embedding_idx],
        param_shapes[models.STUDENT_EMBEDDINGS])

    # reshape flattened assessment embeddings into matrix
    assessment_embeddings = np.reshape(
        param_vals[last_student_embedding_idx:last_assessment_embedding_idx],
        param_shapes[models.ASSESSMENT_EMBEDDINGS])

    if using_graph_prior:
        # reshape flattened concept embeddings into matrix
        concept_embeddings = np.reshape(
            param_vals[last_assessment_bias_idx:],
            param_shapes[models.CONCEPT_EMBEDDINGS])

    # split assessment interactions into students, assessments, outcomes
    (
        student_idxes_for_assessment_ixns,
        assessment_idxes_for_assessment_ixns,
        outcomes_for_assessment_ixns) = assessment_interactions

    # use dummy lesson interactions to get students in temporal process
    student_idxes_for_temporal_process, _, _ = lesson_interactions

    if not using_bias:
        # zero out bias terms, so that they definitely have no effect
        # on the gradient or cost here. this should be done in addition to
        # imposing (0, 0) bounds in the call to scipy.optimize.minimize in est.
        param_vals[last_assessment_embedding_idx:last_assessment_bias_idx] = 0

    # get biases for assessment interactions
    student_biases = np.reshape(
        param_vals[last_assessment_embedding_idx:last_student_bias_idx],
        param_shapes[models.STUDENT_BIASES])[(
        student_idxes_for_assessment_ixns // num_timesteps)][:, None]
    assessment_biases = np.reshape(
        param_vals[last_student_bias_idx:last_assessment_bias_idx],
        param_shapes[models.ASSESSMENT_BIASES])[(
        assessment_idxes_for_assessment_ixns)][:, None]

    # shape outcomes as a column vector
    outcomes = outcomes_for_assessment_ixns[:, None]

    # get the assessment embedding for each assessment interaction
    assessment_embeddings_for_assessment_ixns = assessment_embeddings[\
            assessment_idxes_for_assessment_ixns, :]

    # compute the L2 norm of the assessment embedding for each assessment interaction
    assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
        assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

    # get the student embedding for each assessment interaction
    student_embeddings_for_assessment_ixns = (
        student_embeddings[student_idxes_for_assessment_ixns, :])

    # compute the dot product of the student embedding
    # and assessment embedding for each interaction
    student_dot_assessment = np.einsum(
        'ij, ij->i',
        student_embeddings_for_assessment_ixns,
        assessment_embeddings_for_assessment_ixns)[:, None]

    # compute intermediate quantities for the gradient that get reused
    exp_diff = np.exp(outcomes * (
        assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                assessment_embedding_norms_for_assessment_ixns - student_biases - \
                assessment_biases))
    one_plus_exp_diff = 1 + exp_diff
    mult_diff = outcomes * exp_diff / one_plus_exp_diff

    using_temporal_process = len(student_idxes_for_temporal_process) > 0
    if using_temporal_process:
        # get embeddings of student states resulting from lesson interactions
        curr_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_temporal_process, :]

        # get embeddings of student states prior to lesson interactions
        prev_student_embeddings_for_lesson_ixns = \
                student_embeddings[student_idxes_for_temporal_process - 1, :]

        # compute intermediate quantities for the gradient that get reused
        diffs = curr_student_embeddings_for_lesson_ixns - prev_student_embeddings_for_lesson_ixns \
                + forgetting_penalty_terms
        diffs_over_var = diffs / learning_update_variance
    else:
        diffs = diffs_over_var = 0

    if using_graph_prior:
        # get distance from an assessment embedding to its prior embedding,
        # i.e., the weighted average of the embeddings of the assessment's
        # governing concepts
        assessment_diffs_from_concept_centers = assessment_embeddings - \
                assessment_participation_in_concepts.dot(concept_embeddings)

        # grab the concept dependency graph
        prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
        (
            concept_participation_in_prereqs,
            concept_participation_in_postreqs) = concept_participation_in_prereq_edges

        # get prereq and postreq concept embeddings
        prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
        postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

        postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]
        prereq_dot_postreq = np.einsum(
            'ij, ij->i',
            prereq_concept_embeddings,
            postreq_concept_embeddings)[:, None]

        # intermediate quantity, useful later
        prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

    # compute the gradient w.r.t. student embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
        mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                assessment_embeddings_for_assessment_ixns)
    if using_temporal_process:
        stud_grad_from_temporal_process = curr_student_participation_in_lesson_ixns.dot(
                diffs_over_var)
    else:
        stud_grad_from_temporal_process = 0
    stud_grad_from_norm_regularization = 2 * student_regularization_constant * student_embeddings
    gradient[:last_student_embedding_idx] = (
        stud_grad_from_asmt_ixns + stud_grad_from_temporal_process + \
                stud_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. assessment embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
            mult_diff / assessment_embedding_norms_for_assessment_ixns * (
                    student_embeddings_for_assessment_ixns - \
                            assessment_embeddings_for_assessment_ixns - student_dot_assessment / \
                            np.einsum(
                                'ij, ij->ij',
                                assessment_embedding_norms_for_assessment_ixns,
                                assessment_embedding_norms_for_assessment_ixns) * \
                                        assessment_embeddings_for_assessment_ixns))
    if using_graph_prior:
        asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                assessment_diffs_from_concept_centers
    else:
        asmt_grad_from_graph_regularization = 0
    if using_l1_regularizer:
        asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                assessment_embeddings)
    else:
        asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                assessment_embeddings
    gradient[last_student_embedding_idx:last_assessment_embedding_idx] = (
        asmt_grad_from_asmt_ixns + asmt_grad_from_graph_regularization + \
                asmt_grad_from_norm_regularization).ravel()

    if using_bias:
        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_assessment_embedding_idx:last_student_bias_idx] = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff).ravel()

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_student_bias_idx:last_assessment_bias_idx] = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff).ravel()

    if using_graph_prior:
        # compute the gradient w.r.t. concept embeddings,
        # which is the sum of gradient of the log-likelihood
        # of assessment and lesson interactions and the gradient
        # of the regularization terms
        concept_grad_from_assessments = -concept_participation_in_assessments.dot(
            2 * assessment_diffs_from_concept_centers)

        concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
            postreq_concept_embeddings / postreq_concept_norms)

        concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
            (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / postreq_concept_norms -\
                    2 * prereq_dot_postreq * postreq_concept_embeddings / postreq_concept_norms**3)

        concept_grad_from_norm_regularization = 2 * concept_regularization_constant * \
                concept_embeddings
        gradient[last_assessment_bias_idx:] = (graph_regularization_constant * (
            concept_grad_from_assessments + concept_grad_from_prereqs + \
                    concept_grad_from_postreqs) + concept_grad_from_norm_regularization).ravel()

    cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
    if using_temporal_process:
        cost_from_temporal_process = np.einsum(
                'ij, ij', diffs, diffs) / (2 * learning_update_variance)
    else:
        cost_from_temporal_process = 0
    cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
    if using_l1_regularizer:
        cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                assessment_embeddings).sum()
    else:
        cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)
    if using_graph_prior:
        cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
        cost_from_graph_regularization = graph_regularization_constant * ((
            assessment_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
    else:
        cost_from_concept_regularization = 0
        cost_from_graph_regularization = 0
    cost_from_norm_regularization = cost_from_student_regularization + \
            cost_from_assessment_regularization + cost_from_concept_regularization

    cost_from_regularization = cost_from_norm_regularization + cost_from_graph_regularization
    cost = cost_from_assessment_ixns + cost_from_temporal_process + cost_from_regularization

    return cost, gradient

def with_scipy_with_prereqs(
    param_vals,
    param_shapes,
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer,
    gradient):
    """
    Compute the gradient of the cost function with respect to model parameters

    :param np.array param_vals:
        Flattened, concatenated parameter values

    :param dict[str,tuple] param_shapes:
        A dictionary mapping a parameter's name to the shape of its np.ndarray

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X
        [number of assessment interactions] where a non-zero entry indicates that the student
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X
        [number of assessment interactions] where a non-zero entry indicates that the assessment
        participated in the assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L2 regularization on lesson and assessment embeddings

    :param np.array gradient:
        Placeholder for the flattened gradient

    :rtype: (float,np.array)
    :return:
        The value of the cost function
        (evaluated at the supplied parameter values),
        and the flattened gradient of the cost function
        (evaluated at the supplied parameter values)
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant
    
    # reshape flattened student embeddings into tensor
    student_embeddings = np.reshape(
        param_vals[:last_student_embedding_idx],
        param_shapes[models.STUDENT_EMBEDDINGS])

    # reshape flattened assessment embeddings into matrix
    assessment_embeddings = np.reshape(
        param_vals[last_student_embedding_idx:last_assessment_embedding_idx],
        param_shapes[models.ASSESSMENT_EMBEDDINGS])

    # reshape flattened lesson embeddings into matrix
    lesson_embeddings = np.reshape(
        param_vals[last_assessment_embedding_idx:last_lesson_embedding_idx],
        param_shapes[models.LESSON_EMBEDDINGS])

    # reshape flattened prereq embeddings into matrix
    prereq_embeddings = np.reshape(
        param_vals[last_lesson_embedding_idx:last_prereq_embedding_idx],
        param_shapes[models.PREREQ_EMBEDDINGS])

    if using_graph_prior:
        # reshape flattened concept embeddings into matrix
        concept_embeddings = np.reshape(
            param_vals[last_assessment_bias_idx:],
            param_shapes[models.CONCEPT_EMBEDDINGS])

    # split assessment interactions into students, assessments, outcomes
    (
        student_idxes_for_assessment_ixns,
        assessment_idxes_for_assessment_ixns,
        outcomes_for_assessment_ixns) = assessment_interactions

    # split lesson interactions into students, lessons
    student_idxes_for_lesson_ixns, lesson_idxes_for_lesson_ixns, _ = lesson_interactions

    if not using_bias:
        # zero out bias terms, so that they definitely have no effect
        # on the gradient or cost here. this should be done in addition to
        # imposing (0, 0) bounds in the call to scipy.optimize.minimize in est.
        param_vals[last_prereq_embedding_idx:last_assessment_bias_idx] = 0

    # get biases for assessment interactions
    student_biases = np.reshape(
        param_vals[last_prereq_embedding_idx:last_student_bias_idx],
        param_shapes[models.STUDENT_BIASES])[(
        student_idxes_for_assessment_ixns // num_timesteps)][:, None]
    assessment_biases = np.reshape(
        param_vals[last_student_bias_idx:last_assessment_bias_idx],
        param_shapes[models.ASSESSMENT_BIASES])[assessment_idxes_for_assessment_ixns][:, None]

    # shape outcomes as a column vector
    outcomes = outcomes_for_assessment_ixns[:, None]

    # get the assessment embedding for each assessment interaction
    assessment_embeddings_for_assessment_ixns = \
            assessment_embeddings[assessment_idxes_for_assessment_ixns, :]

    # compute the L2 norm of the assessment embedding for each assessment interaction
    assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
        assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

    # get the student embedding for each assessment interaction
    student_embeddings_for_assessment_ixns = (
        student_embeddings[student_idxes_for_assessment_ixns, :])

    # compute the dot product of the student embedding
    # and assessment embedding for each interaction
    student_dot_assessment = np.einsum(
        'ij, ij->i',
        student_embeddings_for_assessment_ixns,
        assessment_embeddings_for_assessment_ixns)[:, None]

    # compute intermediate quantities for the gradient that get reused
    exp_diff = np.exp(outcomes * (
        assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                assessment_embedding_norms_for_assessment_ixns - student_biases - \
                assessment_biases))
    one_plus_exp_diff = 1 + exp_diff
    mult_diff = outcomes * exp_diff / one_plus_exp_diff

    # get lesson embeddings for lesson interactions
    lesson_embeddings_for_lesson_ixns = lesson_embeddings[lesson_idxes_for_lesson_ixns, :]

    # get lesson prereq embeddings for lesson interactions
    prereq_embeddings_for_lesson_ixns = prereq_embeddings[lesson_idxes_for_lesson_ixns, :]

    # get embeddings of student states resulting from lesson interactions
    curr_student_embeddings_for_lesson_ixns = student_embeddings[student_idxes_for_lesson_ixns, :]

    # get embeddings of student states prior to lesson interactions
    prev_student_embeddings_for_lesson_ixns = \
            student_embeddings[student_idxes_for_lesson_ixns - 1, :]

    # compute the L2 norm of the lesson embedding for each lesson interaction
    prereq_embedding_norms_for_lesson_ixns = np.linalg.norm(
        prereq_embeddings_for_lesson_ixns, axis=1)[:, None]

    # compute the dot product of the student embedding prior
    # to the lesson interaction and the lesson prereq embedding,
    # for each interaction
    prev_student_dot_prereq = np.einsum(
        'ij, ij->i',
        prev_student_embeddings_for_lesson_ixns,
        prereq_embeddings_for_lesson_ixns)[:, None]

    # compute intermediate quantities for the gradient that get reused
    update_exp_diff = np.exp(
        prereq_embedding_norms_for_lesson_ixns - prev_student_dot_prereq / \
                prereq_embedding_norms_for_lesson_ixns)
    update_one_plus_exp_diff = 1 + update_exp_diff
    diffs = curr_student_embeddings_for_lesson_ixns - prev_student_embeddings_for_lesson_ixns - \
            lesson_embeddings_for_lesson_ixns / update_one_plus_exp_diff + forgetting_penalty_terms
    diffs_over_var = diffs / learning_update_variance
    update_mult_diff = np.einsum(
        'ij, ij->i',
        diffs_over_var,
        lesson_embeddings_for_lesson_ixns)[:, None] * update_exp_diff / (
        np.einsum('ij, ij->ij',
            update_one_plus_exp_diff,
            update_one_plus_exp_diff) * prereq_embedding_norms_for_lesson_ixns)

    if using_graph_prior:
        # get distance from an assessment embedding to its prior embedding,
        # i.e., the weighted average of the embeddings of the assessment's
        # governing concepts
        assessment_diffs_from_concept_centers = assessment_embeddings - \
                assessment_participation_in_concepts.dot(concept_embeddings)

        # get distance from a lesson embedding to its prior embedding,
        # i.e., the weighted average of the embeddings of the lesson's
        # governing concepts
        lesson_diffs_from_concept_centers = lesson_embeddings - \
                lesson_participation_in_concepts.dot(concept_embeddings)

        # grab the concept dependency graph
        prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
        concept_participation_in_prereqs, concept_participation_in_postreqs = \
                concept_participation_in_prereq_edges

        # get prereq and postreq concept embeddings
        prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
        postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

        # compute column vector of L2 norms for postreq concept embeddings
        postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]

        # compute the dot product of the prereq concept embedding
        # and postreq concept embedding for each edge in the concept dependency graph
        prereq_dot_postreq = np.einsum(
            'ij, ij->i',
            prereq_concept_embeddings,
            postreq_concept_embeddings)[:, None]

        # intermediate quantity, useful and reusable later
        prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

    # compute the gradient w.r.t. student embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
        mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                assessment_embeddings_for_assessment_ixns)
    stud_grad_from_lesson_ixns = curr_student_participation_in_lesson_ixns.dot(
        diffs_over_var) - prev_student_participation_in_lesson_ixns.dot(
        update_mult_diff * prereq_embeddings_for_lesson_ixns + diffs_over_var)
    stud_grad_from_norm_regularization = 2 * student_regularization_constant * student_embeddings
    gradient[:last_student_embedding_idx] = (
        stud_grad_from_asmt_ixns + stud_grad_from_lesson_ixns + \
                stud_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. assessment embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
            mult_diff / assessment_embedding_norms_for_assessment_ixns * (
                student_embeddings_for_assessment_ixns - assessment_embeddings_for_assessment_ixns\
                        - student_dot_assessment / np.einsum(
                            'ij, ij->ij',
                            assessment_embedding_norms_for_assessment_ixns,
                            assessment_embedding_norms_for_assessment_ixns) * \
                                    assessment_embeddings_for_assessment_ixns))
    if using_graph_prior:
        asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                assessment_diffs_from_concept_centers
    else:
        asmt_grad_from_graph_regularization = 0
    if using_l1_regularizer:
        asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                assessment_embeddings)
    else:
        asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                assessment_embeddings
    gradient[last_student_embedding_idx:last_assessment_embedding_idx] = (
        asmt_grad_from_asmt_ixns + asmt_grad_from_graph_regularization + \
                asmt_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. lesson embeddings,
    # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
    # and the gradient of the regularization terms
    lesson_grad_from_lesson_ixns = -lesson_participation_in_lesson_ixns.dot(
        diffs_over_var / update_one_plus_exp_diff)
    if using_graph_prior:
        lesson_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                lesson_diffs_from_concept_centers
    else:
        lesson_grad_from_graph_regularization = 0
    if using_l1_regularizer:
        lesson_grad_from_norm_regularization = lesson_regularization_constant * np.sign(
                lesson_embeddings)
    else:
        lesson_grad_from_norm_regularization = 2 * lesson_regularization_constant * \
                lesson_embeddings
    gradient[last_assessment_embedding_idx:last_lesson_embedding_idx] = (
        lesson_grad_from_lesson_ixns + lesson_grad_from_graph_regularization + \
                lesson_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. prereq embeddings,
    # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
    # and the gradient of the regularization terms
    prereq_grad_from_lesson_ixns = lesson_participation_in_lesson_ixns.dot(
            update_mult_diff * (prev_student_dot_prereq / np.einsum(
                'ij, ij->ij',
                prereq_embedding_norms_for_lesson_ixns,
                prereq_embedding_norms_for_lesson_ixns) * \
                        prereq_embeddings_for_lesson_ixns - \
                        prev_student_embeddings_for_lesson_ixns + \
                        prereq_embeddings_for_lesson_ixns))
    prereq_grad_from_norm_regularization = 2 * prereq_regularization_constant * prereq_embeddings
    gradient[last_lesson_embedding_idx:last_prereq_embedding_idx] = (
            prereq_grad_from_lesson_ixns + prereq_grad_from_norm_regularization).ravel()

    if using_bias:
        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_prereq_embedding_idx:last_student_bias_idx] = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff).ravel()

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_student_bias_idx:last_assessment_bias_idx] = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff).ravel()

    if using_graph_prior:
        # compute the gradient w.r.t. concept embeddings,
        # which is the sum of gradient of the log-likelihood of assessment
        # and lesson interactions and the gradient of the regularization terms
        concept_grad_from_assessments = -concept_participation_in_assessments.dot(
            2 * assessment_diffs_from_concept_centers)

        concept_grad_from_lessons = concept_participation_in_lessons.dot(
            2 * lesson_diffs_from_concept_centers)

        concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
            postreq_concept_embeddings / postreq_concept_norms)

        concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
            (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / \
                    postreq_concept_norms - 2 * prereq_dot_postreq * postreq_concept_embeddings / \
                    postreq_concept_norms**3)

        gradient[last_assessment_bias_idx:] = graph_regularization_constant * (
            concept_grad_from_assessments + concept_grad_from_lessons + concept_grad_from_prereqs +
            concept_grad_from_postreqs).ravel()

    cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
    cost_from_lesson_ixns = np.einsum('ij, ij', diffs, diffs) / (2 * learning_update_variance)
    cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
    if using_l1_regularizer:
        cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                assessment_embeddings).sum()
        cost_from_lesson_regularization = lesson_regularization_constant * np.absolute(
                lesson_embeddings).sum()
    else:
        cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)
        cost_from_lesson_regularization = lesson_regularization_constant * np.einsum(
                'ij, ij', lesson_embeddings, lesson_embeddings)
    cost_from_prereq_regularization = prereq_regularization_constant * np.einsum(
            'ij, ij', prereq_embeddings, prereq_embeddings)
    if using_graph_prior:
        cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
        cost_from_graph_regularization = graph_regularization_constant * ((
            assessment_diffs_from_concept_centers**2).sum() + (
            lesson_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
    else:
        cost_from_concept_regularization = 0
        cost_from_graph_regularization = 0
    cost_from_norm_regularization = cost_from_student_regularization + \
            cost_from_assessment_regularization + cost_from_lesson_regularization + \
            cost_from_prereq_regularization + cost_from_concept_regularization

    cost_from_ixns = cost_from_assessment_ixns + cost_from_lesson_ixns
    cost_from_regularization = cost_from_norm_regularization + cost_from_graph_regularization
    cost = cost_from_ixns + cost_from_regularization

    return cost, gradient

def with_scipy_without_prereqs(
    param_vals,
    param_shapes,
    assessment_interactions,
    lesson_interactions,
    learning_update_variance,
    forgetting_penalty_terms,
    regularization_constant,
    graph_regularization_constant,
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
    num_timesteps,
    using_bias,
    using_graph_prior,
    using_l1_regularizer,
    gradient):
    """
    Compute the gradient of the cost function with respect to model parameters

    :param np.array param_vals:
        Flattened, concatenated parameter values

    :param dict[str,tuple] param_shapes:
        A dictionary mapping a parameter's name to the shape of its np.ndarray

    :param (np.array,np.array,np.array) assessment_interactions:
        For each assessment interaction, (student_idx, assessment_idx, outcome),
        where outcome is -1 or 1

    :param (np.array,np.array,np.array) lesson_interactions:
        For each lesson interaction, (student_idx, lesson_idx, time_since_previous_interaction)

    :param np.array|float learning_update_variance:
        Variance of the Gaussian learning update. If float, then the variance
        is constant across all interactions. If np.array, then the variance is
        different for each lesson interaction.

    :param np.array|float forgetting_penalty_terms:
        Penalty term for the forgetting effect in the Gaussian learning update.
        If float, then the penalty term is constant across all interactions. If
        np.array, then the penalty is different for each lesson interaction.

    :param (float,float,float,float,float) regularization_constant:
        Coefficients of the regularization terms for (students, assessments,
        lessons, prereqs, concepts)

    :param float graph_regularization_constant:
        Coefficient of the graph regularization term

    :param scipy.sparse.csr_matrix student_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of assessment interactions] where a non-zero entry indicates that the student at a
        specific timestep participated in the assessment interaction

    :param scipy.sparse.csr_matrix student_bias_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique students] X [number of assessment
        interactions] where a non-zero entry indicates that the student participated in the
        assessment interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_assessment_ixns:
        A binary matrix of dimensions [number of unique assessments] X [number of assessment
        interactions] where a non-zero entry indicates that the assessment participated in the
        assessment interaction

    :param scipy.sparse.csr_matrix curr_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the post-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix prev_student_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique students * number of timesteps] X
        [number of lesson interactions] where a non-zero entry indicates that the student at a
        specific timestep was the pre-update student state for the lesson interaction

    :param scipy.sparse.csr_matrix lesson_participation_in_lesson_ixns:
        A binary matrix of dimensions [number of unique lessons] X [number of lesson interactions]
        where a non-zero entry indicates that the lesson participated in the lesson interaction

    :param scipy.sparse.csr_matrix assessment_participation_in_concepts:
        A binary matrix of dimensions [number of unique assessments] X [number of unique concepts],
        where an entry indicates assessment-concept association. Concept associations for a given
        assessment sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix lesson_participation_in_concepts:
        A binary matrix of dimensions [number of unique lessons] X [number of unique concepts],
        where an entry indicates lesson-concept association. Concept associations for a given
        lesson sum to one, i.e., each row sums to one.

    :param scipy.sparse.csr_matrix concept_participation_in_assessments:
        The transpose of assessment_participation_in_concepts

    :param scipy.sparse.csr_matrix concept_participation_in_lessons:
        The transpose of lesson_participation_in_lessons

    :param (np.array,np.array) prereq_edge_concept_idxes:
        (Indices of prereq concepts, Indices of postreq concepts)

    :param (scipy.sparse.csr_matrix,scipy.sparse.csr_matrix) concept_participation_in_prereq_edges:
        The first binary matrix has dimensions [number of unique concepts] X
        [number of prereq edges], where a non-zero entry indicates that the concept is the prereq
        in the edge.

        The second binary matrix has the same dimensions,
        where a non-zero entry indicates that the concept is the postreq in the edge.

    :param int last_student_embedding_idx:
        Index of the last student embedding parameter in the flattened gradient

    :param int last_assessment_embedding_idx:
        Index of the last assessment embedding parameter in the flattened gradient

    :param int last_lesson_embedding_idx:
        Index of the last lesson embedding parameter in the flattened gradient

    :param int last_prereq_embedding_idx:
        Index of the last prereq embedding parameter in the flattened gradient

    :param int last_student_bias_idx:
        Index of the last student bias parameter in the flattened gradient

    :param int last_assessment_bias_idx:
        Index of the last assessment bias parameter in the flattened gradient

    :param int num_timesteps:
        Maximum number of timesteps in a student history, i.e.,
        the output of InteractionHistory.duration()

    :param bool using_bias:
        Including bias terms in the assessment result likelihood

    :param bool using_graph_prior:
        Including the graph regularization term

    :param bool using_l1_regularizer:
        True => use L1 regularization on lesson and assessment embeddings
        False => use L2 regularization on lesson and assessment embeddings

    :param np.array gradient:
        Placeholder for the flattened gradient

    :rtype: (float,np.array)
    :return:
        The value of the cost function
        (evaluated at the supplied parameter values),
        and the flattened gradient of the cost function
        (evaluated at the supplied parameter values)
    """

    # pull regularization constants for different parameters out of tuple
    (
        student_regularization_constant,
        assessment_regularization_constant,
        lesson_regularization_constant,
        prereq_regularization_constant,
        concept_regularization_constant) = regularization_constant
    
    # reshape flattened student embeddings into tensor
    student_embeddings = np.reshape(
        param_vals[:last_student_embedding_idx],
        param_shapes[models.STUDENT_EMBEDDINGS])

    # reshape flattened assessment embeddings into matrix
    assessment_embeddings = np.reshape(
        param_vals[last_student_embedding_idx:last_assessment_embedding_idx],
        param_shapes[models.ASSESSMENT_EMBEDDINGS])

    # reshape flattened lesson embeddings into matrix
    lesson_embeddings = np.reshape(
        param_vals[last_assessment_embedding_idx:last_lesson_embedding_idx],
        param_shapes[models.LESSON_EMBEDDINGS])

    if using_graph_prior:
        # reshape flattened concept embeddings into matrix
        concept_embeddings = np.reshape(
            param_vals[last_assessment_bias_idx:],
            param_shapes[models.CONCEPT_EMBEDDINGS])

    # split assessment interactions into students, assessments, outcomes
    (
        student_idxes_for_assessment_ixns,
        assessment_idxes_for_assessment_ixns,
        outcomes_for_assessment_ixns) = assessment_interactions

    # split lesson interactions into students, lessons
    student_idxes_for_lesson_ixns, lesson_idxes_for_lesson_ixns, _ = lesson_interactions

    if not using_bias:
        # zero out bias terms, so that they definitely have no effect
        # on the gradient or cost here. this should be done in addition to
        # imposing (0, 0) bounds in the call to scipy.optimize.minimize in est.
        param_vals[last_lesson_embedding_idx:last_assessment_bias_idx] = 0

    # get biases for assessment interactions
    student_biases = np.reshape(
        param_vals[last_lesson_embedding_idx:last_student_bias_idx],
        param_shapes[models.STUDENT_BIASES])[(
        student_idxes_for_assessment_ixns // num_timesteps)][:, None]
    assessment_biases = np.reshape(
        param_vals[last_student_bias_idx:last_assessment_bias_idx],
        param_shapes[models.ASSESSMENT_BIASES])[(
        assessment_idxes_for_assessment_ixns)][:, None]

    # shape outcomes as a column vector
    outcomes = outcomes_for_assessment_ixns[:, None]

    # get the assessment embedding for each assessment interaction
    assessment_embeddings_for_assessment_ixns = \
            assessment_embeddings[assessment_idxes_for_assessment_ixns, :]

    # compute the L2 norm of the assessment embedding for each assessment interaction
    assessment_embedding_norms_for_assessment_ixns = np.linalg.norm(
        assessment_embeddings_for_assessment_ixns, axis=1)[:, None]

    # get the student embedding for each assessment interaction
    student_embeddings_for_assessment_ixns = \
            student_embeddings[student_idxes_for_assessment_ixns, :]

    # compute the dot product of the student embedding
    # and assessment embedding for each interaction
    student_dot_assessment = np.einsum(
        'ij, ij->i',
        student_embeddings_for_assessment_ixns,
        assessment_embeddings_for_assessment_ixns)[:, None]

    # compute intermediate quantities for the gradient that get reused
    exp_diff = np.exp(outcomes * (
        assessment_embedding_norms_for_assessment_ixns - student_dot_assessment / \
                assessment_embedding_norms_for_assessment_ixns - student_biases - \
                assessment_biases))
    one_plus_exp_diff = 1 + exp_diff
    mult_diff = outcomes * exp_diff / one_plus_exp_diff

    # get lesson embeddings for lesson interactions
    lesson_embeddings_for_lesson_ixns = lesson_embeddings[lesson_idxes_for_lesson_ixns, :]

    # get embeddings of student states resulting from lesson interactions
    curr_student_embeddings_for_lesson_ixns = student_embeddings[student_idxes_for_lesson_ixns, :]

    # get embeddings of student states prior to lesson interactions
    prev_student_embeddings_for_lesson_ixns = \
            student_embeddings[student_idxes_for_lesson_ixns - 1, :]

    # compute intermediate quantities for the gradient that get reused
    diffs = curr_student_embeddings_for_lesson_ixns - (
        prev_student_embeddings_for_lesson_ixns) - (
        lesson_embeddings_for_lesson_ixns) + forgetting_penalty_terms
    diffs_over_var = diffs / learning_update_variance

    if using_graph_prior:
        # get distance from an assessment embedding to its prior embedding,
        # i.e., the weighted average of the embeddings of the assessment's
        # governing concepts
        assessment_diffs_from_concept_centers = assessment_embeddings - \
                assessment_participation_in_concepts.dot(concept_embeddings)

        # get distance from a lesson embedding to its prior embedding,
        # i.e., the weighted average of the embeddings of the lesson's
        # governing concepts
        lesson_diffs_from_concept_centers = (
            lesson_embeddings) - lesson_participation_in_concepts.dot(
            concept_embeddings)

        # grab the concept dependency graph
        prereq_concept_idxes, postreq_concept_idxes = prereq_edge_concept_idxes
        concept_participation_in_prereqs, concept_participation_in_postreqs = \
                concept_participation_in_prereq_edges

        # get prereq and postreq concept embeddings
        prereq_concept_embeddings = concept_embeddings[prereq_concept_idxes, :]
        postreq_concept_embeddings = concept_embeddings[postreq_concept_idxes, :]

        # compute column vector of L2 norms for postreq concept embeddings
        postreq_concept_norms = np.linalg.norm(postreq_concept_embeddings, axis=1)[:, None]

        # compute the dot product of the prereq concept embedding
        # and postreq concept embedding for each edge in the concept dependency graph
        prereq_dot_postreq = np.einsum(
            'ij, ij->i',
            prereq_concept_embeddings,
            postreq_concept_embeddings)[:, None]

        # intermediate quantity, useful and reusable later
        prereq_edge_diffs = prereq_dot_postreq / postreq_concept_norms - postreq_concept_norms

    # compute the gradient w.r.t. student embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    stud_grad_from_asmt_ixns = -student_participation_in_assessment_ixns.dot(
        mult_diff / assessment_embedding_norms_for_assessment_ixns * \
                assessment_embeddings_for_assessment_ixns)
    stud_grad_from_lesson_ixns = curr_student_participation_in_lesson_ixns.dot(diffs_over_var)
    stud_grad_from_norm_regularization = 2 * student_regularization_constant * student_embeddings
    gradient[:last_student_embedding_idx] = (
        stud_grad_from_asmt_ixns + stud_grad_from_lesson_ixns + \
                stud_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. assessment embeddings,
    # which is the sum of gradient of the log-likelihood of assessment interactions
    # and the gradient of the regularization terms
    asmt_grad_from_asmt_ixns = -assessment_participation_in_assessment_ixns.dot(
        mult_diff / assessment_embedding_norms_for_assessment_ixns * (
            student_embeddings_for_assessment_ixns - assessment_embeddings_for_assessment_ixns - \
                    student_dot_assessment / np.einsum(
                        'ij, ij->ij',
                        assessment_embedding_norms_for_assessment_ixns,
                        assessment_embedding_norms_for_assessment_ixns) * \
                                assessment_embeddings_for_assessment_ixns))
    if using_graph_prior:
        asmt_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                assessment_diffs_from_concept_centers
    else:
        asmt_grad_from_graph_regularization = 0
    if using_l1_regularizer:
        asmt_grad_from_norm_regularization = assessment_regularization_constant * np.sign(
                assessment_embeddings)
    else:
        asmt_grad_from_norm_regularization = 2 * assessment_regularization_constant * \
                assessment_embeddings
    gradient[last_student_embedding_idx:last_assessment_embedding_idx] = (
        asmt_grad_from_asmt_ixns + asmt_grad_from_graph_regularization + \
                asmt_grad_from_norm_regularization).ravel()

    # compute the gradient w.r.t. lesson embeddings,
    # which is the sum of gradient of the log-likelihood of assessment and lesson interactions
    # and the gradient of the regularization terms
    lesson_grad_from_lesson_ixns = -lesson_participation_in_lesson_ixns.dot(diffs_over_var)
    if using_graph_prior:
        lesson_grad_from_graph_regularization = 2 * graph_regularization_constant * \
                lesson_diffs_from_concept_centers
    else:
        lesson_grad_from_graph_regularization = 0
    if using_l1_regularizer:
        lesson_grad_from_norm_regularization = lesson_regularization_constant * np.sign(
                lesson_embeddings)
    else:
        lesson_grad_from_norm_regularization = 2 * lesson_regularization_constant * \
                lesson_embeddings
    gradient[last_assessment_embedding_idx:last_lesson_embedding_idx] = (
        lesson_grad_from_lesson_ixns + lesson_grad_from_graph_regularization + \
                lesson_grad_from_norm_regularization).ravel()

    if using_bias:
        # compute the gradient w.r.t. student biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_lesson_embedding_idx:last_student_bias_idx] = \
                -student_bias_participation_in_assessment_ixns.dot(mult_diff).ravel()

        # compute the gradient w.r.t. assessment biases,
        # which is the sum of gradient of the log-likelihood of assessment interactions
        gradient[last_student_bias_idx:last_assessment_bias_idx] = \
                -assessment_participation_in_assessment_ixns.dot(mult_diff).ravel()

    if using_graph_prior:
        # compute the gradient w.r.t. concept embeddings,
        # which is the sum of gradient of the log-likelihood of assessment
        # and lesson interactions and the gradient of the regularization terms
        concept_grad_from_assessments = -concept_participation_in_assessments.dot(
            2 * assessment_diffs_from_concept_centers)

        concept_grad_from_lessons = concept_participation_in_lessons.dot(
            2 * lesson_diffs_from_concept_centers)

        concept_grad_from_prereqs = concept_participation_in_prereqs.dot(
            postreq_concept_embeddings / postreq_concept_norms)

        concept_grad_from_postreqs = concept_participation_in_postreqs.dot(
            (prereq_concept_embeddings - 2 * postreq_concept_embeddings) / postreq_concept_norms \
                    - 2 * prereq_dot_postreq * postreq_concept_embeddings / \
                    postreq_concept_norms**3)

        concept_grad_from_norm_regularization = 2 * concept_regularization_constant * \
                concept_embeddings
        gradient[last_assessment_bias_idx:] = (graph_regularization_constant * (
            concept_grad_from_assessments + concept_grad_from_lessons + \
                    concept_grad_from_prereqs + concept_grad_from_postreqs) + \
                    concept_grad_from_norm_regularization).ravel()

    cost_from_assessment_ixns = np.einsum('ij->', np.log(one_plus_exp_diff))
    cost_from_lesson_ixns = np.einsum('ij, ij', diffs, diffs) / (2 * learning_update_variance)
    cost_from_student_regularization = student_regularization_constant * np.einsum(
            'ij, ij', student_embeddings, student_embeddings)
    if using_l1_regularizer:
        cost_from_assessment_regularization = assessment_regularization_constant * np.absolute(
                assessment_embeddings).sum()
        cost_from_lesson_regularization = lesson_regularization_constant * np.absolute(
                lesson_embeddings).sum()
    else:
        cost_from_assessment_regularization = assessment_regularization_constant * np.einsum(
                'ij, ij', assessment_embeddings, assessment_embeddings)
        cost_from_lesson_regularization = lesson_regularization_constant * np.einsum(
                'ij, ij', lesson_embeddings, lesson_embeddings)
    if using_graph_prior:
        cost_from_concept_regularization = concept_regularization_constant * np.einsum(
                'ij, ij', concept_embeddings, concept_embeddings)
        cost_from_graph_regularization = graph_regularization_constant * ((
            assessment_diffs_from_concept_centers**2).sum() + (
            lesson_diffs_from_concept_centers**2).sum() + prereq_edge_diffs.sum())
    else:
        cost_from_concept_regularization = 0
        cost_from_graph_regularization = 0
    cost_from_norm_regularization = cost_from_student_regularization + \
            cost_from_assessment_regularization + cost_from_lesson_regularization + \
            cost_from_concept_regularization

    cost_from_ixns = cost_from_assessment_ixns + cost_from_lesson_ixns
    cost_from_regularization = cost_from_norm_regularization + cost_from_graph_regularization
    cost = cost_from_ixns + cost_from_regularization

    return cost, gradient

def get_grad(
    using_scipy=True,
    using_lessons=True,
    using_prereqs=True):
    """
    Select the appropriate gradient and cost function evaluator
    for a model configuration

    :param bool using_scipy: Using scipy.optimize.minize for optimization
    :param bool using_lessons: Including lessons in the embedding model
    :param bool using_prereqs: Including lesson prereqs in the embedding model
    :rtype: function
    :return: A function that takes current parameter values
        as input, and outputs the gradient of the cost function
        with respect to those parameters
    """

    if using_scipy:
        if using_lessons:
            if using_prereqs:
                return with_scipy_with_prereqs
            else:
                return with_scipy_without_prereqs
        else:
            return with_scipy_without_lessons
    else:
        if using_lessons:
            if using_prereqs:
                return without_scipy_with_prereqs
            else:
                return without_scipy_without_prereqs
        else:
            return without_scipy_without_lessons

