"""
Module for storing and manipulating interaction histories

@author Siddharth Reddy <sgr45@cornell.edu>
"""
from __future__ import division

from abc import abstractmethod
from collections import defaultdict, namedtuple
import datetime as dt
import logging
import random
import re
import uuid

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class Interaction(object):
    """
    An interaction
    """

    def __init__(
        self,
        student_id=None,
        module_id=None,
        timestep=None,
        timestamp=None,
        outcome=None,
        duration=None,
        module_type=None):
        """
        Initialize interaction object

        QUESTION: These can all be None. What happens when they're None? Why isn't that
            reflected in the typing?

        :param str student_id: A student_id
        :param str module_id: A module_id
        :param int timestep: A discretized timestep (strictly positive)
        :param bool|None outcome: Pass/fail for assessment interactions,
            None for lesson interactions
        :param int|None duration: A duration (in seconds),
            None if duration is not available
        """
        if timestep <= 0:
            raise ValueError('Invalid timestep!')
        if duration is not None and duration < 0:
            raise ValueError('Invalid duration!')

        self.student_id = student_id
        self.module_id = module_id
        self.timestep = timestep
        self.timestamp = timestamp
        self.duration = duration
        self.outcome = outcome
        self.module_type = module_type


class LessonInteraction(Interaction):
    """
    An interaction with a lesson module
    """

    MODULETYPE = 'lesson'

    def __init__(
        self,
        student_id=None,
        module_id=None,
        timestep=None,
        timestamp=None,
        duration=None):
        """
        Initialize an interaction object that references a lesson

        :param str student_id: A student_id
        :param str module_id: A module_id
        :param dt.datetime|pd.Datetime timestamp: A timestamp
        :param int timestep: A discretized timestep (strictly positive)
        :param int|None duration: A duration (in seconds),
            None if duration is N/A
        """
        super(LessonInteraction, self).__init__(
            student_id=student_id,
            module_id=module_id,
            timestep=timestep,
            timestamp=timestamp,
            duration=duration,
            outcome=None,
            module_type=self.MODULETYPE)


class AssessmentInteraction(Interaction):
    """
    An interaction with an assessment module
    """

    MODULETYPE = 'assessment'

    def __init__(
        self,
        student_id=None,
        module_id=None,
        timestep=None,
        timestamp=None,
        outcome=None,
        duration=None):
        """
        Initialize an interaction object that references an assessment

        :param str student_id: A student_id
        :param str module_id: A module_id
        :param dt.datetime|pd.Datetime timestamp: A timestamp
        :param int timestep: A discretized timestep (strictly positive)
        :param bool|None outcome: Pass/fail for assessment interactions,
            None for lesson interactions
        :param int|None duration: A duration (in seconds),
            None if duration is N/A
        """
        super(AssessmentInteraction, self).__init__(
            student_id=student_id,
            module_id=module_id,
            timestep=timestep,
            timestamp=timestamp,
            duration=duration,
            outcome=outcome,
            module_type=self.MODULETYPE)


class InteractionHistory(object):
    """
    Class for an interaction history
    """

    def __init__(self, data, sort_by_timestep=False):
        """
        Initialize interaction history object

        :param pd.DataFrame data: An interaction history dataframe with the following columns::

            outcome : bool|None
            student_id : str
            module_id : str
            module_type : str
            timestep : int
            timestamp : datetime.datetime
            duration : int

        :param bool sort_by_timestep:
            True => sort interactions in increasing order of timestep
        """
        self.data = data.sort('timestep', axis=0, inplace=False) if sort_by_timestep else data

        # optional columns
        if 'time_since_previous_interaction' not in self.data.columns:
            self.data['time_since_previous_interaction'] = np.nan
        if 'duration' not in self.data.columns:
            self.data['duration'] = np.nan
        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = np.nan

        # need to add 1 since internal time starts at zero
        self._duration = self.data['timestep'].max() + 1

        # dict[str, int]
        # student_id -> student index
        self._student_idx = None

        # dict[str, int]
        # assessment_id -> assessment index
        self._assessment_idx = None

        # dict[str, int]
        # lesson_id -> lesson index
        self._lesson_idx = None

        # dict[int, str]
        # student index -> student_id
        self._student_inv_idx = None

        # dict[int, str]
        # assessment index -> assessment_id
        self._assessment_inv_idx = None

        # dict[int, str]
        # lesson index -> lesson_id
        self._lesson_inv_idx = None

        # dict[str, int]
        # student_id -> timestep
        self._timestep_of_last_interaction = {}

        self.compute_idx_maps()

    def compute_idx_maps(
        self,
        student_idx=None,
        assessment_idx=None,
        lesson_idx=None):
        """
        Fill in self._*idx to map string IDs to indices
        and self._*_inv_idx to map indices to string IDs

        :param dict[str, int]|None student_idx:
            A dictionary mapping student_id to student_idx
        :param dict[str, int]|None assessment_idx:
            A dictionary mapping assessment_id to assessment_idx
        :param dict[str, int]|None lesson_idx:
            A dictionary mapping lesson_id to lesson_idx
        """

        student_ids = self.data['student_id'].unique()
        assessment_ids = self.data['module_id'][(
            self.data['module_type']==AssessmentInteraction.MODULETYPE)].unique()
        lesson_ids = self.data['module_id'][(
            self.data['module_type']==LessonInteraction.MODULETYPE)].unique()

        buildidx = lambda s: {x: i for i, x in enumerate(s)}
        self._student_idx = buildidx(student_ids) if student_idx is None else student_idx
        self._assessment_idx = buildidx(assessment_ids) if assessment_idx is None else assessment_idx
        self._lesson_idx = buildidx(lesson_ids) if lesson_idx is None else lesson_idx

        build_inv_idx = lambda d: {v: k for k, v in d.iteritems()}
        self._student_inv_idx = build_inv_idx(self._student_idx)
        self._assessment_inv_idx = build_inv_idx(self._assessment_idx)
        self._lesson_inv_idx = build_inv_idx(self._lesson_idx)

    def squash_timesteps(self, num_checkpoints=10):
        """
        Squash timesteps for consecutive assessment interactions,
        i.e., timestep should only increment when a student works on a lesson

        :param int num_checkpoints: Number of checkpoints to log during iteration over students
        """

        num_students = self.num_students()
        checkpoint_len = num_students // num_checkpoints

        df = self.data.groupby('student_id')

        for student_idx, student_id in enumerate(self.iter_students()):
            if checkpoint_len > 0 and student_idx % checkpoint_len == 0:
                _logger.debug(
                    "Processed student %d of %d", student_idx, num_students)
            timestep = 1
            for idx, interaction in df.get_group(student_id).iterrows():
                module_type = interaction['module_type']
                if module_type == LessonInteraction.MODULETYPE:
                    timestep += 1

                self.data.loc[idx, 'timestep'] = timestep

        # need to add 1 since internal time starts at zero
        self._duration = self.data['timestep'].max() + 1

    def split_interactions_by_type(
        self,
        filtered_history=None):
        """
        QUESTION: The return type of this function is hideous. Can it be made into some
            sort of class something something?

        Split history into assessment interactions and lesson interactions

        Useful for model parameter estimation

        :param pd.DataFrame|None filtered_history: A filtered interaction history
        :rtype: ((np.ndarray, np.ndarray, np.ndarray),
            (np.ndarray, np.ndarray, np.ndarray), dict[str, int])
        :return A tuple of assessment interaction components
            (np.array(student_idxes), np.array(module_idxes), np.array(outcomes)),
            lesson interaction components (np.array(student_idxes),
                np.array(module_idxes), np.array(times_since_previous_interaction)),
            and a dictionary mapping student_id to the timestep of the student's last interaction
        """

        df = filtered_history if filtered_history is not None else self.data

        num_timesteps = self.duration()

        def df_to_unique_timesteps(dataframe):
            """ Convert a dataframe's timesteps so that only one event happens per timestep """
            output = dataframe['student_id'].apply(self.idx_of_student_id).values
            return output * num_timesteps + dataframe['timestep'].values

        assessment_df = df[df['module_type']==AssessmentInteraction.MODULETYPE]
        assessment_timesteps = df_to_unique_timesteps(assessment_df)
        assessment_idx = assessment_df['module_id'].apply(self.idx_of_assessment_id)

        # Convert bools to +/- 1
        assessment_outcome = assessment_df['outcome'] * 2 - 1

        assessment_interactions = (assessment_timesteps, assessment_idx, assessment_outcome)

        lesson_df = df[df['module_type']==LessonInteraction.MODULETYPE]
        lesson_timesteps = df_to_unique_timesteps(lesson_df)
        lesson_idx = lesson_df['module_id'].apply(self.idx_of_lesson_id).values
        lesson_time_delta = lesson_df['time_since_previous_interaction'].values
        lesson_interactions = (lesson_timesteps, lesson_idx, lesson_time_delta)

        timestep_of_last_interaction = df.groupby('student_id')['timestep'].max() + 1
        num_students = self.num_students()

        return (assessment_interactions,
                lesson_interactions,
                timestep_of_last_interaction.to_dict())

    def id_of_student_idx(self, student_idx):
        """
        QUESTION: Why do you need these functions instead of using the dicts directly?
            Is it for pandas? In that case, you can always use the .map function instead
            of .apply

        :param int student_idx: An index of a student
        :rtype: str
        :return: The id of the student at that idx
        """
        return self._student_inv_idx[student_idx]

    def id_of_assessment_idx(self, assessment_idx):
        """
        :param int assessment_idx: An index of a assessment
        :rtype: str
        :return: The id of the assessment at that idx
        """
        return self._assessment_inv_idx[assessment_idx]

    def id_of_lesson_idx(self, lesson_idx):
        """
        :param int lesson_idx: An index of a lesson
        :rtype: str
        :return: The id of the lesson at that idx
        """
        return self._lesson_inv_idx[lesson_idx]

    def idx_of_student_id(self, student_id):
        """
        Get student index of a student id

        :param str student_id: A student id
        :rtype: int
        :return: The student's index
        """
        return self._student_idx[student_id]

    def idx_of_assessment_id(self, assessment_id):
        """
        Get assessment index of an assessment id

        :param str assessment_id: An assessment id
        :rtype: int
        :return: The assessment's index
        """

        return self._assessment_idx[assessment_id]

    def idx_of_lesson_id(self, lesson_id):
        """
        Get lesson index of a lesson_id

        :param str lesson_id: A lesson id
        :rtype: int
        :return: The lesson's index
        """
        return self._lesson_idx[lesson_id]

    def duration(self):
        """
        Get the total duration of the interaction history

        :rtype: int
        :return: The total duration
        """
        return self._duration

    def assessment_pass_rate(self, assessment_id, ending_timestep=None):
        """
        Get the pass rate for an assessment

        :param str assessment_id: An assessment id
        :param int|None ending_timestep:
            Only look at interactions before and at this timestep
            If None, then look at all interactions

        :rtype: float
        :return: Pass rate
        """
        if ending_timestep is not None and (
            ending_timestep <= 0 or ending_timestep >= self.duration()):
            raise ValueError('Invalid timestep!')

        if ending_timestep is None:
            df = self.data
        else:
            df = self.data[self.data['timestep']<=ending_timestep]
        outcomes = df['outcome'][df['module_id']==assessment_id]
        if len(outcomes) == 0:
            raise ValueError('No interactions for assessment {}'.format(assessment_id))

        num_passes = outcomes.sum()
        return num_passes / len(outcomes)

    def num_students(self):
        """
        :rtype: int
        :return: Total number of students
        """
        return len(self._student_idx)

    def num_assessments(self):
        """
        :rtype: int
        :return: Total number of assessments
        """
        return len(self._assessment_idx)

    def num_lessons(self):
        """
        :rtype: int
        :return: Total number of lessons
        """
        return len(self._lesson_idx)

    def iter_students(self):
        """
        :rtype: dictionary-keyiterator
        :return: Iterator over student_ids
        """
        return self._student_idx.iterkeys()

    def iter_assessments(self):
        """
        :rtype: dictionary-keyiterator
        :return: Iterator over assessment_ids
        """

        return self._assessment_idx.iterkeys()

    def iter_lessons(self):
        """
        :rtype: dictionary-keyiterator
        :return: Iterator over lesson_ids
        """
        return self._lesson_idx.iterkeys()

    def module_sequence_of_student(self, student_id):
        """
        Get sequence of modules for student

        :param str student_id: A student_id
        :rtype: pd.Series
        :return: A sequence of module_ids
        """
        return self.data['module_id'][self.data['student_id']==student_id]
