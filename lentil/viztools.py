"""
Module for visualizing skill embeddings

@author Siddharth Reddy <sgr45@cornell.edu>
"""

import logging

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from . import models


_logger = logging.getLogger(__name__)


def plot_embedding(
    model,
    timestep=-1,
    show_students=True,
    show_assessments=True,
    show_lessons=None,
    show_prereqs=None,
    show_concepts=None,
    show_student_ids=False,
    show_assessment_ids=False,
    show_lesson_ids=False,
    show_concept_ids=False,
    id_padding_x=0.01,
    id_padding_y=0.01,
    alpha=0.5,
    size=20,
    title='',
    show_legend=True,
    force_invariant_axis_limits=True,
    axis_limit_padding=0.1,
    show_pass_rates=False,
    x_axis_limits=None,
    y_axis_limits=None):
    """
    Plot students, assessments, lessons, and prereqs
    in a two-dimensional skill embedding

    Students, assessments, prereqs = points
    Lessons = vectors

    See nb/synthetic_experiments.ipynb for example invocations

    :param EmbeddingModel model: A skill embedding model
    :param int timestep: A timestep. By default, timestep=-1 => latest snapshot
    :param float id_padding_x: Padding between object and id along x-axis
    :param float id_padding_y: Padding between object and id along y-axis
    :param float alpha: Alpha level for scatterplot points' color
    :param int size: Size of scatterplot points
    :param str|None title: Title of plot
    :param bool show_legend:
        True => show legend in upper left corner
        False => do not show legend

    :param bool force_invariant_axis_limits:
        True => plot will have same axes limits regardless of timestep,
        False => plot may have different axes limits depending on timestep

    :param float axis_limit_padding:
        Padding for axis limits (to prevent points from being stuck
        at the edges of the plot)

    :param bool show_pass_rates:
        True => color assessments by pass rate,
        False => don't color assessments

    :param list[int,int]|None x_axis_limits: [x_min, x_max]
    :param list[int,int]|None y_axis_limits: [y_min, y_max]
    """
    if model.embedding_dimension != 2:
        raise ValueError('Invalid embedding dimension!')
    if timestep<-1 or timestep>=model.history.duration():
        raise ValueError('Invalid timestep!')
    if size<=0:
        raise ValueError('Invalid scatterplot point size!')
    if axis_limit_padding<0:
        raise ValueError('Invalid axis limit padding!')
    if show_lessons is None:
        show_lessons = model.using_lessons
    if show_prereqs is None:
        show_prereqs = model.using_prereqs
    if show_lessons and not model.using_lessons:
        raise ValueError(
            'Cannot show lessons because model does not use lessons!')
    if show_prereqs and not model.using_prereqs:
        raise ValueError(
            'Cannot show prereqs because model does not use prereqs!')
    if show_concepts and not model.using_graph_prior:
        raise ValueError(
            'Cannot show concepts because model does not use a graph prior!')
    if show_student_ids and not show_students:
        raise ValueError('Cannot show student_ids without students!')
    if show_assessment_ids and not show_assessments:
        raise ValueError('Cannot show assessment_ids without assessments!')
    if show_lesson_ids and not show_lessons and not show_prereqs:
        raise ValueError('Cannot show lesson_ids without lessons and/or prereqs!')
    if show_pass_rates and not show_assessments:
        raise ValueError('Cannot show pass rates without assessments!')
    if show_concept_ids and not show_concepts:
        raise ValueError('Cannot show concept_ids without concepts!')


    if show_pass_rates and model.history.num_students() > 1:
        _logger.warning('Showing pass rates for more than one student!')

    _, ax = plt.subplots()

    if show_students:
        student_embeddings_x = model.student_embeddings[:, 0, timestep]
        student_embeddings_y = model.student_embeddings[:, 1, timestep]
        ax.scatter(
            student_embeddings_x, student_embeddings_y,
            alpha=alpha, marker='o', s=size, label='student')

        if show_student_ids:
            for student_id in model.history.iter_students():
                student_idx = model.history.idx_of_student_id(student_id)
                student_x = student_embeddings_x[student_idx]
                student_y = student_embeddings_y[student_idx]
                student_id_x = student_x + id_padding_x
                student_id_y = student_y + id_padding_y
                ax.annotate(student_id, xy=(
                    student_x, student_y), xytext=(
                    student_id_x, student_id_y))

    if show_assessments:
        assessment_embeddings_x = model.assessment_embeddings[:, 0]
        assessment_embeddings_y = model.assessment_embeddings[:, 1]
        if show_pass_rates:
            num_assessments = model.history.num_assessments()
            pass_rates = [model.history.assessment_pass_rate(
                model.history.id_of_assessment_idx(
                    i), timestep if timestep!=-1 else None) for i in xrange(
                num_assessments)]
            ax.scatter(
                assessment_embeddings_x,
                assessment_embeddings_y,
                c=pass_rates,
                alpha=alpha,
                marker='s',
                s=size,
                label='assessment',
                cmap=matplotlib.cm.cool)
        else:
            ax.scatter(
                assessment_embeddings_x,
                assessment_embeddings_y,
                alpha=alpha,
                marker='s',
                s=size,
                label='assessment')

        if show_assessment_ids:
            for assessment_id in model.history.iter_assessments():
                assessment_idx = model.history.idx_of_assessment_id(assessment_id)
                assessment_x = assessment_embeddings_x[assessment_idx]
                assessment_y = assessment_embeddings_y[assessment_idx]
                assessment_id_x = assessment_x + id_padding_x
                assessment_id_y = assessment_y + id_padding_y
                ax.annotate(assessment_id, xy=(
                    assessment_x, assessment_y), xytext=(
                    assessment_id_x, assessment_id_y))

    if show_concepts:
        concept_embeddings_x = model.concept_embeddings[:, 0]
        concept_embeddings_y = model.concept_embeddings[:, 1]
        ax.scatter(
            concept_embeddings_x,
            concept_embeddings_y,
            alpha=alpha,
            marker='^',
            s=size,
            label='concept')

        if show_concept_ids:
            for concept_id, concept_idx in model.graph.idx_of_concept_id.iteritems():
                concept_x = concept_embeddings_x[concept_idx]
                concept_y = concept_embeddings_y[concept_idx]
                concept_id_x = concept_x + id_padding_x
                concept_id_y = concept_y + id_padding_y
                ax.annotate(concept_id, xy=(
                    concept_x, concept_y), xytext=(
                    concept_id_x, concept_id_y))

    if show_lessons:
        if model.using_prereqs and show_prereqs:
            prereq_embeddings_x = model.prereq_embeddings[:, 0]
            prereq_embeddings_y = model.prereq_embeddings[:, 1]
        else:
            prereq_embeddings_x = prereq_embeddings_y = [0] * (
                model.history.num_lessons())
        lesson_embeddings_x = model.lesson_embeddings[:, 0]
        lesson_embeddings_y = model.lesson_embeddings[:, 1]
        ax.quiver(
            prereq_embeddings_x, prereq_embeddings_y,
            lesson_embeddings_x, lesson_embeddings_y, pivot='tail')

        if show_lesson_ids:
            for lesson_id in model.history.iter_lessons():
                lesson_idx = model.history.idx_of_lesson_id(lesson_id)
                lesson_x = prereq_embeddings_x[lesson_idx] if model.using_prereqs else 0
                lesson_y = prereq_embeddings_y[lesson_idx] if model.using_prereqs else 0
                lesson_id_x = lesson_x + id_padding_x
                lesson_id_y = lesson_y + id_padding_y
                ax.annotate(lesson_id, xy=(
                    lesson_x, lesson_y), xytext=(
                    lesson_id_x, lesson_id_y))

    if show_legend:
        ax.legend(loc='upper left')

    if force_invariant_axis_limits:
        x = []
        y = []
        if show_students:
            x += np.unique(model.student_embeddings[:, 0, :]).tolist()
            y += np.unique(model.student_embeddings[:, 1, :]).tolist()
        if show_assessments:
            x += np.unique(model.assessment_embeddings[:, 0]).tolist()
            y += np.unique(model.assessment_embeddings[:, 1]).tolist()
        if show_lessons:
            x += np.unique(model.lesson_embeddings[:, 0] + (
                model.prereq_embeddings[:, 0] if show_prereqs else 0)).tolist()
            y += np.unique(model.lesson_embeddings[:, 1] + (
                model.prereq_embeddings[:, 1] if show_prereqs else 0)).tolist()
        if show_concepts:
            x += np.unique(model.concept_embeddings[:, 0]).tolist()
            y += np.unique(model.concept_embeddings[:, 1]).tolist()
        ax.set_xlim([min(x)-axis_limit_padding, max(x)+axis_limit_padding])
        ax.set_ylim([min(y)-axis_limit_padding, max(y)+axis_limit_padding])

    if x_axis_limits is not None:
        ax.set_xlim(x_axis_limits)
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)

    if title is None:
        title = 'Latent Skill Space'
    ax.set_title(title)

    ax.set_xlabel('Skill 1')
    ax.set_ylabel('Skill 2')

    plt.show()

