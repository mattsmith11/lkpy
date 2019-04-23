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
from . import Predictor

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

    def __init__(self, regularization=(.5, 1.0)):
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

        coeff_row = np.array([], dtype=np.int32)
        coeff_col = np.array([], dtype=np.int32)
        coeff_values = np.array([], dtype=np.float64)

        rmat_copy = rmat.to_scipy().copy()

        for item in range(rmat.ncols):
            # Create an ElasticNet optimization function
            opt_model = ElasticNet(alpha=self.alpha,l1_ratio=self.l_1_ratio,positive=True,fit_intercept=True,copy_X=False)

            sp_rmat = rmat_copy.copy()
            item_col = sp_rmat.getcol(item)

            if (item % 100) == 0: _logger.info('[%s] computed coefficients for %s items', self._timer, item)
                
            # Zero out the column of the item before optimizing to prevent the model from optimizing for the item itself
            sp_rmat[sp_rmat[:, item].nonzero()[0], item] = 0

            opt_model.fit(sp_rmat, item_col.todense())
            assert opt_model.coef_[item] == 0

            # Indexes of coefficient array with positive values
            sparse_coeff_coo = opt_model.sparse_coef_.tocoo() 
            # Add coefficients with proper indexes for sparse matrix
            coeff_row = np.append(coeff_row, np.full(sparse_coeff_coo.nnz, item))
            coeff_col = np.append(coeff_col, sparse_coeff_coo.col)
            coeff_values = np.append(coeff_values, sparse_coeff_coo.data)

        #_logger.info('[%s] 1 %s - 2 %s', self._timer, coeff_row, coeff_row2)
        #_logger.info('[%s] 1 %s - 2 %s', self._timer, coeff_col, coeff_col2)
        #_logger.info('[%s] 1 %s - 2 %s', self._timer, coeff_values, coeff_values2)

        _logger.warn('[%s] completed calculating coefficients for %s items', self._timer, rmat.ncols)

        # Create sparse coefficient matrix 
        self.coefficients_ = CSR.from_coo(coeff_row, coeff_col, coeff_values, (len(iidx), len(iidx))).to_scipy()

        self.user_index_ = uidx
        self.item_index_ = iidx
        self.ratings_matrix_ = rmat

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): NOT SUPPORTED

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        _logger.debug('predicting %d item(s) for user %s', len(items), user)

        if ratings is not None:
            _logger.debug('SLIM does not support ratings fit at predict time')
            raise NotImplementedError()
            
        upos = self.user_index_.get_loc(user)
        ipos = self.item_index_.get_indexer(items)

        if -1 is upos:
            _logger.warn("The requested user was not in the original dataset")

        if -1 in ipos:
            _logger.warn("Some item ratings requested are for items missing from fit data set")

        ipos = ipos[ipos >= 0]

        _logger.debug('Items %s have positions %s in the coefficient matrix', items, ipos )

        urow = self.ratings_matrix_.row(upos)
        urow = np.reshape(urow, (-1, 1)).transpose()

        raw_scores = (urow @ self.coefficients_[:, ipos])[0]
        indexed_scores = {}
        raw_scores_index = 0
        for i in ipos:
            _logger.debug('Predicted score for %s is %s', i, raw_scores[raw_scores_index])
            indexed_scores[self.item_index_[i]] = raw_scores[raw_scores_index]
            raw_scores_index += 1
        return pd.Series(indexed_scores)

    def __str__(self):
        return 'SLIM(regularization_one={}, regularization_two={})'.format(self.regularization_one, self.regularization_two)