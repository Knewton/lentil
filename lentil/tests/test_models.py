"""
Module for unit tests that check if estimated embeddings
for toy examples satisfy basic intuitions
@author Siddharth Reddy <sgr45@cornell.edu>
07/15/15
"""

import unittest
import logging

import pandas as pd
import numpy as np

from lentil import models
from lentil import est
from lentil import toy

logging.basicConfig()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class TestModels(unittest.TestCase):

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
        recovered positive skill gains for L1, and ``correctly" arranged
        students and assessments in the latent space. Initially, Carter
        fails both assessments, so his skill level is behind the requirements
        of both assessments. Lee passes A1 but fails A2, so his skill
        level is beyond the requirement for A1, but behind the requirement
        for A2. In an effort to improve their results, Lee and Carter
        complete lesson L1 and retake both assessments. Now Carter passes
        A1, but still fails A2, so his new skill level is ahead of the
        requirements for A1 but behind the requirements for A2. Lee
        passes both assessments, so his new skill level exceeds the requirements
        for A1 and A2. This clear difference in results before completing
        lesson L1 and after completing the lesson implies that L1 had a
        positive effect on Lee and Carter's skill levels, hence the non-zero
        skill gain vector recovered for L1.
        """

        history = toy.get_1d_embedding_history()

        embedding_dimension = 1

        model = models.EmbeddingModel(
            history,
            embedding_dimension,
            using_lessons=True,
            using_prereqs=False,
            using_bias=False,
            learning_update_variance_constant=0.5)

        gradient_descent_kwargs = {
            'using_adagrad' : False,
            'rate' : 0.1,
            'debug_mode_on' : False
        }

        using_scipy_configs = [True, False]
        for using_scipy in using_scipy_configs:
            estimator = est.EmbeddingMAPEstimator(
                regularization_constant=1e-6,
                gradient_descent_kwargs=gradient_descent_kwargs,
                using_scipy=using_scipy,
                verify_gradient=False,
                debug_mode_on=False)

            model.fit(estimator)

            lee = model.student_embeddings[model.history.idx_of_student_id('Lee'), 0, 1:]
            carter = model.student_embeddings[model.history.idx_of_student_id('Carter'), 0, 1:]

            a1 = model.assessment_embeddings[model.history.idx_of_assessment_id('A1'), 0]
            a2 = model.assessment_embeddings[model.history.idx_of_assessment_id('A2'), 0]

            self.assertTrue((carter[0] < a1).all())
            self.assertTrue((a1 < lee[0]).all())
            self.assertTrue((lee[0] < a2).all())

            self.assertTrue((a1 < carter[1]).all())
            self.assertTrue((carter[1] < a2).all())
            self.assertTrue((a2 < lee[1]).all())

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

        model = models.EmbeddingModel(
            history,
            embedding_dimension,
            using_prereqs=False,
            using_lessons=False,
            using_bias=False,
            learning_update_variance_constant=0.5)

        gradient_descent_kwargs = {
            'using_adagrad' : False,
            'rate' : 0.1,
            'debug_mode_on' : False
        }

        using_scipy_configs = [True, False]
        for using_scipy in using_scipy_configs:
            estimator = est.EmbeddingMAPEstimator(
                regularization_constant=1e-6,
                gradient_descent_kwargs=gradient_descent_kwargs,
                using_scipy=using_scipy,
                verify_gradient=False,
                debug_mode_on=False)

            model.fit(estimator)

            mclovin = model.student_embeddings[model.history.idx_of_student_id('McLovin'), :, 1]
            fogell = model.student_embeddings[model.history.idx_of_student_id('Fogell'), :, 1]
            seth = model.student_embeddings[model.history.idx_of_student_id('Seth'), :, 1]
            evan = model.student_embeddings[model.history.idx_of_student_id('Evan'), :, 1]

            a1 = model.assessment_embeddings[model.history.idx_of_assessment_id('A1'), :]
            a2 = model.assessment_embeddings[model.history.idx_of_assessment_id('A2'), :]

            self.assertTrue((mclovin > model.assessment_embeddings[:, :]).all())
            self.assertTrue((fogell <= model.assessment_embeddings[:, :]).all())

            eps = 1.0
            self.assertTrue((seth >= a1-eps).all() and (seth > a1).any())
            self.assertTrue((seth < a2).any())

            self.assertTrue((evan >= a2-eps).all() and (evan > a2).any())
            self.assertTrue((evan < a1).any())

    def test_independent_lessons(self):
        """
        We replicate the setting in Figure \ref{fig:superbad}, then add two
        new students Slater and Michaels, and two new lesson modules L1
        and L2. Slater is initially identical to Evan, while Michaels is
        initially identical to Seth. Slater reads lesson L1, then passes
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

        model = models.EmbeddingModel(
            history,
            embedding_dimension,
            using_prereqs=False,
            using_lessons=True,
            using_bias=False,
            learning_update_variance_constant=0.5)

        gradient_descent_kwargs = {
            'using_adagrad' : False,
            'rate' : 0.1,
            'debug_mode_on' : False
        }

        using_scipy_configs = [True, False]
        for using_scipy in using_scipy_configs:
            estimator = est.EmbeddingMAPEstimator(
                regularization_constant=1e-6,
                gradient_descent_kwargs=gradient_descent_kwargs,
                using_scipy=using_scipy,
                verify_gradient=False,
                debug_mode_on=False)

            model.fit(estimator)

            mclovin = model.student_embeddings[model.history.idx_of_student_id('McLovin'), :, 1]
            fogell = model.student_embeddings[model.history.idx_of_student_id('Fogell'), :, 1]
            seth = model.student_embeddings[model.history.idx_of_student_id('Seth'), :, 1]
            evan = model.student_embeddings[model.history.idx_of_student_id('Evan'), :, 1]
            slater = model.student_embeddings[model.history.idx_of_student_id('Slater'), :, 1:]
            michaels = model.student_embeddings[model.history.idx_of_student_id('Michaels'), :, 1:]

            a1 = model.assessment_embeddings[model.history.idx_of_assessment_id('A1'), :]
            a2 = model.assessment_embeddings[model.history.idx_of_assessment_id('A2'), :]

            l1 = model.lesson_embeddings[model.history.idx_of_lesson_id('L1'), :]
            l2 = model.lesson_embeddings[model.history.idx_of_lesson_id('L2'), :]

            self.assertTrue((mclovin > model.assessment_embeddings[:, :]).all())
            self.assertTrue((fogell <= model.assessment_embeddings[:, :]).all())

            eps = 1.0
            self.assertTrue((seth >= a1-eps).all() and (seth > a1).any())
            self.assertTrue((seth < a2).any())
            self.assertTrue((slater[:, 0] >= a1-eps).all() and (slater[0] > a1).any())
            self.assertTrue((slater[:, 0] < a2).any())

            self.assertTrue((evan >= a2-eps).all() and (evan > a2).any())
            self.assertTrue((evan < a1).any())
            self.assertTrue((michaels[:, 0] >= a2-eps).all() and (michaels[0] > a2).any())
            self.assertTrue((michaels[:, 0] < a1).any())

            self.assertTrue((slater[:, 1] > model.assessment_embeddings[:, :]).all())
            self.assertTrue((michaels[:, 1] > model.assessment_embeddings[:, :]).all())

            max_ind = lambda s: max(range(len(s)), key=lambda k: s[k])

            self.assertTrue(max_ind(l1) == max_ind(a2))
            self.assertTrue(max_ind(l2) == max_ind(a1))


    def test_lesson_prereqs(self):
        """
        We replicate the setting in Figure \ref{fig:superbad}, then add a new
        assessment module A3 and a new lesson module L1. All students
        initially fail assessment A3, then read lesson L1, after which
        McLovin passes A3 while everyone else still fails A3. The key
        observation here is that McLovin is the only student who initially
        satisfies the prerequisites for L1, so he is the only student who
        realizes significant gains.
        """

        history = toy.get_lesson_prereqs_history()

        embedding_dimension = 2

        model = models.EmbeddingModel(
            history,
            embedding_dimension,
            using_lessons=True,
            using_prereqs=True,
            using_bias=False,
            learning_update_variance_constant=0.5)

        gradient_descent_kwargs = {
            'using_adagrad' : False,
            'rate' : 0.01,
            'ftol' : 1e-4,
            'debug_mode_on' : False
        }

        using_scipy_configs = [True, False]
        for using_scipy in using_scipy_configs:
            estimator = est.EmbeddingMAPEstimator(
                regularization_constant=0.001,
                gradient_descent_kwargs=gradient_descent_kwargs,
                using_scipy=using_scipy,
                verify_gradient=False,
                debug_mode_on=False)

            model.fit(estimator)

            mclovin = model.student_embeddings[model.history.idx_of_student_id('McLovin'), :, 1:]
            fogell = model.student_embeddings[model.history.idx_of_student_id('Fogell'), :, 1:]
            seth = model.student_embeddings[model.history.idx_of_student_id('Seth'), :, 1:]
            evan = model.student_embeddings[model.history.idx_of_student_id('Evan'), :, 1:]

            a1 = model.assessment_embeddings[model.history.idx_of_assessment_id('A1'), :]
            a2 = model.assessment_embeddings[model.history.idx_of_assessment_id('A2'), :]
            a3 = model.assessment_embeddings[model.history.idx_of_assessment_id('A3'), :]

            l1 = model.lesson_embeddings[model.history.idx_of_lesson_id('L1'), :]
            q1 = model.prereq_embeddings[model.history.idx_of_lesson_id('L1'), :]

            self.assertTrue((mclovin[:, 0] > a1).all())
            self.assertTrue((mclovin[:, 0] > a2).all())
            self.assertTrue((mclovin[:, 0] < a3).any())
            for i in xrange(3):
                self.assertTrue((fogell[:, 0] <= model.assessment_embeddings[i, :]).any())
                self.assertTrue((fogell[:, 1] <= model.assessment_embeddings[i, :]).any())

            eps = models.ANTI_SINGULARITY_LOWER_BOUND
            for i in xrange(2):
                self.assertTrue((seth[:, i] >= a1 - eps).all() and (seth[:, i] > a1 - eps).any())
                self.assertTrue((seth[:, i] < a2).any())
                self.assertTrue((seth[:, i] < a3).any())

            for i in xrange(2):
                self.assertTrue((evan[:, i] >= a2 - eps).all() and (evan[:, i] > a2 - eps).any())
                self.assertTrue((evan[:, i] < a1).any())
                self.assertTrue((evan[:, i] < a3).any())

            _logger.debug(mclovin[:, 1])
            _logger.debug(model.assessment_embeddings)
            self.assertTrue((mclovin[:, 1] > model.assessment_embeddings[:, :]).all())

            # prereq satisfaction term
            prereq_sat = lambda s: model.prereq_weight(s, q1)

            self.assertTrue(prereq_sat(mclovin[0]) > prereq_sat(fogell[0]))
            self.assertTrue(prereq_sat(mclovin[0]) > prereq_sat(seth[0]))
            self.assertTrue(prereq_sat(mclovin[0]) > prereq_sat(evan[0]))

if __name__ == '__main__':
    unittest.main()
