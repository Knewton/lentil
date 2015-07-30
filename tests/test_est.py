"""
Module for unit tests that check if parameter estimation converges for toy examples

@author Siddharth Reddy <sgr45@cornell.edu>
"""

import copy
import unittest
import logging

import pandas as pd
import numpy as np

from lentil import models
from lentil import est
from lentil import toy


logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class TestEstimators(unittest.TestCase):

    def setUp(self):
        # fixes random initializations of parameters
        # before parameter estimation
        np.random.seed(1997)

    def tearDown(self):
        pass

    def test_1d_embedding(self):
        """
        A one-dimensional embedding, where a single latent skill is enough
        to explain the data. The key observation here is that the model
        recovered positive skill gains for $L_1$, and "correctly" arranged
        students and assessments in the latent space. Initially, Carter
        fails both assessments, so his skill level is behind the requirements
        of both assessments. Lee passes A1 but fails A2, so his skill
        level is beyond the requirement for A1, but behind the requirement
        for A2. In an effort to improve their results, Lee and Carter
        complete lesson $L_1$ and retake both assessments. Now Carter passes
        A1, but still fails A2, so his new skill level is ahead of the
        requirements for A1 but behind the requirements for A2. Lee
        passes both assessments, so his new skill level exceeds the requirements
        for A1 and A2. This clear difference in results before completing
        lesson $L_1$ and after completing the lesson implies that $L_1$ had a
        positive effect on Lee and Carter's skill levels, hence the non-zero
        skill gain vector recovered for $L_1$.
        """

        history = toy.get_1d_embedding_history()

        embedding_dimension = 1

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-6

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            model = models.EmbeddingModel(
                history,
                embedding_dimension,
                using_lessons=True,
                using_prereqs=False,
                using_bias=False,
                using_l1_regularizer=using_l1_regularizer,
                learning_update_variance_constant=0.5)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    def test_assessment_grid(self):
        """
        A two-dimensional grid of assessments and a single student
        somewhere in the middle of it
        """

        embedding_kwargs = {
            'embedding_dimension' : 2,
            'using_lessons' : False,
            'using_prereqs' : False,
            'using_bias' : False,
            'learning_update_variance_constant' : 0.5
        }

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-3

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            embedding_kwargs.update({'using_l1_regularizer' : using_l1_regularizer})
            model = toy.get_assessment_grid_model(embedding_kwargs)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    def test_independent_assessments(self):
        """
        A two-dimensional embedding, where an intransitivity in assessment
        results requires more than one latent skill to explain. The key
        observation here is that the assessments are embedded on two different
        axes, meaning they require two completely independent skills. This
        makes sense, since student results on A1 are uncorrelated with
        results on A2. Fogell fails both assessments, so his skill levels
        are behind the requirements for A1 and A2. McLovin passes both
        assessments, so his skill levels are beyond the requirements for A1
        and A2. Evan and Seth are each able to pass one assessment but not
        the other. Since the assessments have independent requirements, this
        implies that Evan and Seth have independent skill sets
        (i.e. Evan has enough of skill 2 to pass A2 but not enough of
        skill 1 to pass A1, and Seth has enough of skill 1 to pass A1
        but not enough of skill 2 to pass A2).
        """

        history = toy.get_independent_assessments_history()

        embedding_dimension = 2

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-6

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            model = models.EmbeddingModel(
                history,
                embedding_dimension,
                using_prereqs=False,
                using_lessons=False,
                using_bias=False,
                using_l1_regularizer=using_l1_regularizer,
                learning_update_variance_constant=0.5)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    def test_independent_lessons(self):
        """
        We replicate the setting in test_independent_assessments, then add two
        new students Slater and Michaels, and two new lesson modules $L_1$
        and L2. Slater is initially identical to Evan, while Michaels is
        initially identical to Seth. Slater reads lesson $L_1$, then passes
        assessments A1 and A2. Michaels reads lesson L2, then passes
        assessments A1 and A2. The key observation here is that the skill
        gain vectors recovered for the two lesson modules are orthogonal,
        meaning they help students satisfy completely independent skill
        requirements. This makes sense, since initially Slater was lacking
        in Skill 1 while Michaels was lacking in Skill 2, but after completing
        their lessons they passed their assessments, showing that they gained
        from their respective lessons what they were lacking initially.
        """

        history = toy.get_independent_lessons_history()

        embedding_dimension = 2

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-6

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            model = models.EmbeddingModel(
                history,
                embedding_dimension,
                using_prereqs=False,
                using_lessons=True,
                using_bias=False,
                using_l1_regularizer=using_l1_regularizer,
                learning_update_variance_constant=0.5)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    def test_lesson_prereqs(self):
        """
        We replicate the setting in test_independent_assessments, then add a new
        assessment module A3 and a new lesson module L1. All students
        initially fail assessment A3, then read lesson L1, after which
        McLovin passes A3 while everyone else still fails A3. The key
        observation here is that McLovin is the only student who initially
        satisfies the prerequisites for L1, so he is the only student who
        realizes significant gains.
        """

        history = toy.get_lesson_prereqs_history()

        embedding_dimension = 2

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-6

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            model = models.EmbeddingModel(
                history,
                embedding_dimension,
                using_prereqs=False,
                using_lessons=True,
                using_bias=False,
                using_l1_regularizer=using_l1_regularizer,
                learning_update_variance_constant=0.5)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    def test_using_bias(self):
        """
        Try using bias terms in assessment result likelihood
        """

        history = toy.get_1d_embedding_history()

        embedding_dimension = 2

        estimator = est.EmbeddingMAPEstimator(
            regularization_constant=1e-6,
            using_scipy=True,
            verify_gradient=True,
            debug_mode_on=False)

        eps = 1e-6

        using_l1_regularizer_configs = [True, False]
        for using_l1_regularizer in using_l1_regularizer_configs:
            model = models.EmbeddingModel(
                history,
                embedding_dimension,
                using_prereqs=False,
                using_lessons=True,
                using_bias=False,
                using_l1_regularizer=using_l1_regularizer,
                learning_update_variance_constant=0.5)

            model.fit(estimator)

            self.assertTrue(estimator.fd_err < eps)

    # TODO: add unit tests for tv_luv_model, forgetting_model, using_graph_prior=True,
    # and using_lessons=False for temporal process on student
    
if __name__ == '__main__':
    unittest.main()

