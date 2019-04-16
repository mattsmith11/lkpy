"""
SLIM algorithm
"""

import logging

import pandas as pd
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as spla
import sklearn as skl
from sklearn.linear_model import SGDRegressor, ElasticNet


from lenskit import util
from .. import check
from ..matrix import CSR, sparse_ratings
from . import Recommender, CandidateSelector

_logger = logging.getLogger(__name__)


class SLIM(Predictor):
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

    def __init__(self, regularization=(1.0, 2.0)):
        if isinstance(regularization, tuple):
            self.regularization = regularization
            self.l_1_regularization, self.l_2_regularization = regularization
        else:
            self.regularization = regularization
            self.l_1_regularization = regularization
            self.l_2_regularization = regularization

        check.check_value(self.l_1_regularization >= 0, "l_1 norm regularization value {} must be nonnegative",
                          self.l_1_regularization)
        check.check_value(self.l_2_regularization >= 0, "l_2 norm regularization {} must be nonnegative",
                          self.l_2_regularization)

        # Calculating alpha using the two regularization values
        self.alpha = self.l_1_regularization + self.l_2_regularization
        self.l_1_ratio = self.l_1_regularization / self.alpha

        self.opt_model = ElasticNet(alpha=self.alpha,l1_ratio=self.l_1_ratio,positive=True,fit_intercept=True,copy_X=False)

    def fit(self, data):
        """
        Run the optimization problem to learn W.

        Args:
            data (DataFrame): a data frame of ratings. Must have at least `user`,
                              `item`, and `rating` columns.

        Returns:
            SLIM: the fit slim algorithm object.
        """
        self._timer = util.Stopwatch()

        rmat, uidx, iidx = sparse_ratings(data)

        coeff_row = []
        coeff_col = []
        coeff_values = []

        for item in range(rmat.ncols):
            sp_rmat = rmat.to_scipy()
            item_col = sp_rmat.getcol(item)

            if (item % 100) == 0: _logger.info('[%s] computing coefficients for item %s', self._timer, item)
            # Zero out the column of the item before optimizing to prevent the model
            sp_rmat[sp_rmat[:, item].nonzero()[0], item] = 0
            #assert sp_rmat.shape == rmat.to_scipy().shape

            self.opt_model.fit(sp_rmat, item_col.toarray())
            #self.opt_model.coef_[item] = 0
            assert self.opt_model.coef_[item] == 0

            # Remove negative coefficients to enforce positive relations and 0s for sparsity
            for coefficient, index in enumerate(self.opt_model.coef_):
                if coefficient > 0:
                    coeff_row.append(item)
                    coeff_col.append(index)
                    coeff_values.append(coefficient)


        _logger.info('[%s] completed calculating coefficients', self._timer)

        # Create sparse coefficient matrix 
        row_ind = iidx.get_indexer(coeff_row).astype(np.int32)
        col_ind = iidx.get_indexer(coeff_col).astype(np.int32)
        coeff_vals = np.require(coeff_values, np.float64)
        self.coefficients_ = CSR.from_coo(row_ind, col_ind, coeff_vals, (len(iidx), len(iidx))).to_scipy()

        self.user_index_ = uidx
        self.item_index_ = iidx

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