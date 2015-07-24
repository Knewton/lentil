"""
Module for generating synthetic interaction histories
@author Siddharth Reddy <sgr45@cornell.edu>
01/07/15
"""

import logging

import numpy as np
import pandas as pd

from lentil import datatools
from lentil import models


_logger = logging.getLogger(__name__)


def sample_synthetic_model_and_history(
    sample_students,
    sample_assessments,
    sample_lessons,
    sample_prereqs,
    sample_interactions,
    embedding_kwargs):
    """
    Sample a synthetic skill embedding and interaction history

    It is up to you to make sure that sample_assessments and sample_prereqs
    adhere to the bounds given in models.EmbeddingModel.anti_singularity_lower_bounds

    :param function sample_students: A function that outputs a student tensor
    :param function sample_assessments: A function that outputs a assessment matrix
    :param function sample_lessons: A function that outputs a lesson matrix
    :param function sample_prereqs: A function that outputs a lesson prereq matrix
    :param function sample_interactions: A function that takes a models.EmbeddingModel
        as input and outputs a pd.DataFrame
    :param dict[str, object] embedding_kwargs: Parameters to pass to models.EmbeddingModel constructor
    :rtype: EmbeddingModel
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