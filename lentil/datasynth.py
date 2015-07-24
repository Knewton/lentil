"""
Module for generating synthetic interaction histories

@author Siddharth Reddy <sgr45@cornell.edu>
"""

import logging

import numpy as np

from . import datatools
from . import models


def sample_synthetic_model_and_history(
    sample_students,
    sample_assessments,
    sample_interactions,
    sample_lessons=None,
    sample_prereqs=None,
    **embedding_kwargs):
    """
    Sample a synthetic skill embedding and interaction history

    It is up to you to make sure that sample_assessments and sample_prereqs
    adhere to the bounds given in :py:class:`models.EmbeddingModel`'s anti_singularity_lower_bounds

    QUESTION: Where can I learn the formats of these student tensors (in particular, it reads here
        like you expect these to be called multiple times, but you don't)? Why do I need these to
        be functions instead of a history? What is expected?

    :param function sample_students: A function that outputs a student tensor
    :param function sample_assessments: A function that outputs a assessment matrix
    :param function sample_lessons: A function that outputs a lesson matrix
    :param function sample_prereqs: A function that outputs a lesson prereq matrix
    :param function sample_interactions: A function that takes a models.EmbeddingModel
        as input and outputs a pandas DataFrame
    :param dict[str, object] embedding_kwargs: Parameters to pass to the
        :py:class:`models.EmbeddingModel` constructor
    :rtype: models.EmbeddingModel
    :return: A skill model with a synthetic interaction history
    """

    model = models.EmbeddingModel(None, **embedding_kwargs)

    model.student_embeddings = np.maximum(
        model.anti_singularity_lower_bounds[models.STUDENT_EMBEDDINGS],
        sample_students())
    model.assessment_embeddings = np.maximum(
        model.anti_singularity_lower_bounds[models.ASSESSMENT_EMBEDDINGS],
        sample_assessments())

    lesson_embeddings = sample_lessons()
    if lesson_embeddings is not None:
        model.lesson_embeddings = np.maximum(
            model.anti_singularity_lower_bounds[models.LESSON_EMBEDDINGS],
            lesson_embeddings)

    prereq_embeddings = sample_prereqs()
    if prereq_embeddings is not None:
        model.prereq_embeddings = np.maximum(
            model.anti_singularity_lower_bounds[models.PREREQ_EMBEDDINGS],
            prereq_embeddings)

    model.history = datatools.InteractionHistory(sample_interactions(model))

    return model
