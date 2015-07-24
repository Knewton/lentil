"""
Module for constructing toy interaction histories
@author Siddharth Reddy <sgr45@cornell.edu>
04/05/15
"""

import math
import random

import pandas as pd
import numpy as np

from lentil import datatools
from lentil import datasynth

def get_1d_embedding_history():
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

    data = []

    data.append(
        {'module_id' : 'A1',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : True,
        'student_id' : 'Lee',
        'timestep' : 1})
    data.append(
        {'module_id' : 'A1',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : False,
        'student_id' : 'Carter',
        'timestep' : 1})
    data.append(
        {'module_id' : 'A2',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : False,
        'student_id' : 'Lee',
        'timestep' : 2})
    data.append(
        {'module_id' : 'A2',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : False,
        'student_id' : 'Carter',
        'timestep' : 2})

    data.append(
        {'module_id' : 'L1',
        'module_type' : datatools.LessonInteraction.MODULETYPE,
        'outcome' : None,
        'student_id' : 'Lee',
        'timestep' : 3})
    data.append(
        {'module_id' : 'L1',
        'module_type' : datatools.LessonInteraction.MODULETYPE,
        'outcome' : None,
        'student_id' : 'Carter',
        'timestep' : 3})

    data.append(
        {'module_id' : 'A1',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : True,
        'student_id' : 'Lee',
        'timestep' : 4})

    data.append(
        {'module_id' : 'A1',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : True,
        'student_id' : 'Carter',
        'timestep' : 4})
    data.append(
        {'module_id' : 'A2',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : True,
        'student_id' : 'Lee',
        'timestep' : 5})

    data.append(
        {'module_id' : 'A2',
        'module_type' : datatools.AssessmentInteraction.MODULETYPE,
        'outcome' : False,
        'student_id' : 'Carter',
        'timestep' : 5})

    history = datatools.InteractionHistory(pd.DataFrame(data))
    history.squash_timesteps()

    return history

def get_assessment_grid_model(
    embedding_dimension=2,
    num_assessments=25,
    num_attempts=10):
    """
    A two-dimensional grid of assessments and a single student
    somewhere in the middle of it
    """

    id_of_assessment_idx = lambda idx: 'A' + str(idx + 1)

    def sample_students():
        """
        Fixed at (0.5, 0.5)
        """
        duration = num_assessments * num_attempts + 1
        S = np.zeros((1,
                      embedding_dimension,
                      duration))
        S.fill(0.5)
        return S

    def sample_assessments():
        """
        Uniform grid from (0,0) to (1,1),
        excluding (1,1)
        """
        A = np.zeros((num_assessments,
                      embedding_dimension))
        grid_length = int(math.sqrt(num_assessments))
        for i in xrange(grid_length):
            for j in xrange(grid_length):
                A[i*grid_length+j, 0] = 1 / grid_length * i
                A[i*grid_length+j, 1] = 1 / grid_length * j
        return A

    def sample_lessons():
        """
        No lessons
        """
        return None

    def sample_prereqs():
        """
        No lesson prereqs
        """
        return None

    def sample_interactions(model):
        """
        student works on assessment 1 [num_attempts] times
        student works on assessment 2 [num_attempts] times
        .
        .
        .
        student works on assessment [num_assessments] [num_attempts] times
        """
        data = []

        student_id = 'Carl'
        student = model.student_embeddings[0, :, 0]
        student_bias = 0
        for i in xrange(num_assessments):
            assessment_id = id_of_assessment_idx(i)
            assessment = model.assessment_embeddings[i, :]
            assessment_bias = 0
            pass_likelihood = math.exp(
                model.assessment_outcome_log_likelihood_helper(
                    student, assessment, student_bias, assessment_bias, 1))
            for j in xrange(num_attempts):
                timestep = 1+i*num_attempts + j
                outcome = random.random() < pass_likelihood

                data.append(
                    {'module_id' : assessment_id,
                    'module_type' : datatools.AssessmentInteraction.MODULETYPE,
                    'outcome' : outcome,
                    'student_id' : student_id,
                    'timestep' : timestep})

        return pd.DataFrame(data)

    embedding_kwargs = {
        'embedding_dimension' : embedding_dimension,
        'using_lessons' : False,
        'using_prereqs' : False,
        'using_bias' : False,
        'learning_update_variance_constant' : 0.5
    }

    model = datasynth.sample_synthetic_model_and_history(
        sample_students,
        sample_assessments,
        sample_lessons,
        sample_prereqs,
        sample_interactions,
        embedding_kwargs)

    num_students = model.student_embeddings.shape[0]
    num_assessments = model.assessment_embeddings.shape[0]
    model.student_biases = np.zeros(num_students)
    model.assessment_biases = np.zeros(num_assessments)

    assessment_idx_map = {id_of_assessment_idx(i):i for i in xrange(num_assessments)}
    model.history.compute_idx_maps(assessment_idx=assessment_idx_map)

    model.history.squash_timesteps()

    return model

def get_independent_assessments_history():
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

    data = []

    def complete_assessment(module_id, student_id, outcome, ixn_idx):
        data.append(
            {'module_id' : module_id,
            'module_type' : datatools.AssessmentInteraction.MODULETYPE,
            'outcome' : outcome,
            'student_id' : student_id,
            'timestep' : ixn_idx})

    complete_assessment('A1', 'McLovin', True, 1)
    complete_assessment('A2', 'McLovin', True, 2)

    complete_assessment('A1', 'Fogell', False, 1)
    complete_assessment('A2', 'Fogell', False, 2)

    complete_assessment('A1', 'Seth', True, 1)
    complete_assessment('A2', 'Seth', False, 2)

    complete_assessment('A1', 'Evan', False, 1)
    complete_assessment('A2', 'Evan', True, 2)

    history = datatools.InteractionHistory(pd.DataFrame(data))
    history.squash_timesteps()

    return history

def get_independent_lessons_history():
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

    data = []

    def complete_lesson(lesson_id, student_id, timestep):
        data.append(
            {'module_id' : lesson_id,
            'module_type' : datatools.LessonInteraction.MODULETYPE,
            'outcome' : None,
            'student_id' : student_id,
            'timestep' : timestep})

    def complete_assessment(assessment_id, student_id, outcome, start_time):
        data.append(
            {'module_id' : assessment_id,
            'module_type' : datatools.AssessmentInteraction.MODULETYPE,
            'outcome' : outcome,
            'student_id' : student_id,
            'timestep' : start_time})

    complete_assessment('A1', 'McLovin', True, 1)
    complete_assessment('A2', 'McLovin', True, 2)

    complete_assessment('A1', 'Fogell', False, 1)
    complete_assessment('A2', 'Fogell', False, 2)

    complete_assessment('A1', 'Seth', True, 1)
    complete_assessment('A2', 'Seth', False, 2)

    complete_assessment('A1', 'Evan', False, 1)
    complete_assessment('A2', 'Evan', True, 3)

    complete_assessment('A1', 'Slater', True, 1)
    complete_assessment('A2', 'Slater', False, 1)

    complete_assessment('A1', 'Michaels', False, 1)
    complete_assessment('A2', 'Michaels', True, 1)

    complete_lesson('L1', 'Slater', 4)
    complete_lesson('L2', 'Michaels', 4)

    complete_assessment('A1', 'Slater', True, 5)
    complete_assessment('A2', 'Slater', True, 6)

    complete_assessment('A1', 'Michaels', True, 5)
    complete_assessment('A2', 'Michaels', True, 6)

    history = datatools.InteractionHistory(pd.DataFrame(data))
    history.squash_timesteps()

    return history

def get_lesson_prereqs_history():
    """
    We replicate the setting in Figure \ref{fig:superbad}, then add a new
    assessment module A3 and a new lesson module L1. All students
    initially fail assessment A3, then read lesson L1, after which
    McLovin passes A3 while everyone else still fails A3. The key
    observation here is that McLovin is the only student who initially
    satisfies the prerequisites for L1, so he is the only student who
    realizes significant gains.
    """

    data = []

    def complete_assessment(assessment_id, student_id, outcome, j):
        data.append(
            {'module_id' : assessment_id,
            'module_type' : datatools.AssessmentInteraction.MODULETYPE,
            'outcome' : outcome,
            'student_id' : student_id,
            'timestep' : j})

    def complete_lesson(lesson_id, student_id, timestep):
        data.append(
            {'module_id' : lesson_id,
            'module_type' : datatools.LessonInteraction.MODULETYPE,
            'outcome' : None,
            'student_id' : student_id,
            'timestep' : timestep})

    complete_assessment('A1', 'McLovin', True, 1)
    complete_assessment('A2', 'McLovin', True, 2)

    complete_assessment('A1', 'Fogell', False, 1)
    complete_assessment('A2', 'Fogell', False, 2)

    complete_assessment('A1', 'Seth', True, 1)
    complete_assessment('A2', 'Seth', False, 2)

    complete_assessment('A1', 'Evan', False, 1)
    complete_assessment('A2', 'Evan', True, 3)

    complete_assessment('A3', 'McLovin', False, 4)
    complete_assessment('A3', 'Fogell', False, 4)
    complete_assessment('A3', 'Seth', False, 4)
    complete_assessment('A3', 'Evan', False, 4)

    complete_lesson('L1', 'McLovin', 5)
    complete_lesson('L1', 'Fogell', 5)
    complete_lesson('L1', 'Seth', 5)
    complete_lesson('L1', 'Evan', 5)

    complete_assessment('A3', 'McLovin', True, 6)

    complete_assessment('A3', 'Fogell', False, 6)
    complete_assessment('A3', 'Seth', False, 6)
    complete_assessment('A3', 'Evan', False, 6)

    history = datatools.InteractionHistory(pd.DataFrame(data))
    history.squash_timesteps()

    return history
