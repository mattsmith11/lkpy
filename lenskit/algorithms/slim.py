"""
SLIM algorithm
"""

import logging

import pandas as pd
import numpy as np

from .. import check
from ..matrix import CSR, sparse_ratings
from . import Recommender, CandidateSelector

_logger = logging.getLogger(__name__)


class SLIM(Recommender):
    """
    A user based collaborative filtering Top-N recommendation algorithm. That implements the following
    predictor algorithm:

    .. math::
       s(u,i) = 

    where :math:`u_i` is items purchased by the user, :math:'w_j' is a vector
    of aggregation coefficients. The coefficient vector is a column in the n-sized sparse
    W matrix which is the result of the following optimization problem:

    .. math::
       \\begin{align*}
       
       \\end{align*}

    Args:
        regularization(double or tuple):
            Regularization factor that is applied to the l_1 and l_2 norms in the optimization problem.
            If a tuple of 2 numbers is provided, the regularization factors will
            be applied to the l_1 and l_2 norms respectively.

    Attributes:
        l_1_regularization(double): The l_1 regularization factor.
        l_2_regularization(double): The l_2 regularization factor
    """

    def __init__(self, regularization=1.0):
        if isinstance(regularization, tuple):
            self.regularization = regularization
            self.l_1_regularization, self.l_2_regularization = regularization
        else:
            self.regularization = regularization
            self.l_1_regularization = regularization
            self.l_2_regularization = regularization

        check.check_value(self.regularization_one >= 0, "l_1 norm regularization value {} must be nonnegative",
                          self.regularization_one)
        check.check_value(self.regularization_two >= 0, "l_2 norm regularization {} must be nonnegative",
                          self.regularization_two)

    def fit(self, data):
        """
        Run the optimization problem to learn W.

        Args:
            data (DataFrame): a data frame of ratings. Must have at least `user`,
                              `item`, and `rating` columns.

        Returns:
            SLIM: the fit slim algorithm object.
        """

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): the user's ratings (indexed by item id); if
                                 provided, will be used to recompute the user's
                                 bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        return None

    def __str__(self):
        return 'SLIM(regularization_one={}, regularization_two={})'.format(self.regularization_one, self.regularization_two)