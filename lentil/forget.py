"""
Module for models of offline learning and forgetting

@author Siddharth Reddy <sgr45@cornell.edu>
"""

from abc import abstractmethod

import numpy as np


class TimeVaryingLUVModel(object):
    """
    Superclass for models of time-varying learning update variance
    """

    @abstractmethod
    def learning_update_variances(self, times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction

        :rtype: np.ndarray
        :return: A list of penalty terms that get subtracted
            from the means of Gaussian learning updates
        """
        pass


class LinearLUVModel(TimeVaryingLUVModel):
    """
    A model of learning update variances that increases variance
    linearly with log(time elapsed since previous interaction)
    """

    def __init__(self, alpha, beta):
        """
        Initialize a linear model of time-varying learning update variance

        :param float alpha: Coefficient, which controls how sensitive variance is
            to log(time since previous interaction)
        
        :param float beta: Offset, which controls the baseline variance
        """

        self.alpha = alpha
        self.beta = beta

    def learning_update_variances(
        self,
        times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction
        
        :rtype: np.array
        :return: A list of Gaussian learning update variances,
            one for each lesson interaction
        """

        return self.alpha * np.log(times_since_prev_ixn_for_lesson_ixns+1) + self.beta


class LogisticLUVModel(TimeVaryingLUVModel):
    """
    A model of learning update variances that increases variance
    as a function of log(time elapsed since previous interaction)
    passed through the logistic function
    """

    def __init__(self, alpha, beta):
        """
        Initialize a linear model of time-varying learning update variance

        :param float alpha: Coefficient, which controls how sensitive variance is
            to log(time since previous interaction)
        
        :param float beta: Offset, which controls the baseline variance
        """

        self.alpha = alpha
        self.beta = beta

    def learning_update_variances(
        self,
        times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction
        
        :rtype: np.array
        :return: A list of Gaussian learning update variances,
            one for each lesson interaction
        """

        return self.beta / (1 + np.exp(-self.alpha * np.log(
            times_since_prev_ixn_for_lesson_ixns+1)))


class ForgettingModel(object):
    """
    Superclass for models of the forgetting effect
    """

    @abstractmethod
    def penalty_terms(self, times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction

        :rtype: np.ndarray
        :return: A list of penalty terms that get subtracted
            from the means of Gaussian learning updates,
            one for each lesson interaction
        """
        pass


class LinearForgettingModel(ForgettingModel):
    """
    A model of the forgetting effect that increases learning penalties
    linearly with the log(time elapsed since the previous interaction)
    """

    def __init__(self, alpha, beta):
        """
        Initialize a linear forgetting model

        :param float alpha: Coefficient, which controls how sensitive the forgetting penalty is
            to log(time since previous interaction)
        
        :param float beta: Offset, which controls the baseline variance
        """

        self.alpha = alpha
        self.beta = beta

    def penalty_terms(
        self,
        times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction
        
        :rtype: np.array
        :return: A list of penalty terms that get subtracted
            from the means of Gaussian learning updates,
            one for each lesson interaction
        """

        return -(self.alpha * np.log(times_since_prev_ixn_for_lesson_ixns+1) + self.beta)


class LogisticForgettingModel(ForgettingModel):
    """
    A model of the forgetting effect that increases learning penalties
    as a function of the log(time elapsed since the previous interaction)
    passed through the logistic function
    """

    def __init__(self, alpha, beta):
        """
        Initialize a logistic forgetting model

        :param float alpha: Coefficient, which controls how sensitive the forgetting penalty is
            to log(time since previous interaction)
        
        :param float beta: Offset, which controls the baseline forgetting penalty
        """

        self.alpha = alpha
        self.beta = beta

    def penalty_terms(
        self,
        times_since_prev_ixn_for_lesson_ixns):
        """
        :param np.array times_since_prev_ixn_for_lesson_ixns:
            Time since the previous interaction, for each lesson interaction
        
        :rtype: np.array
        :return: A list of penalty terms that get subtracted
            from the means of Gaussian learning updates
        """

        return -self.beta / (1 + np.exp(-self.alpha * np.log(
            times_since_prev_ixn_for_lesson_ixns+1)))

