"""
SLIM algorithm
"""

import logging
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import scipy.sparse as sps
import scipy.sparse.linalg as spla
import sklearn as skl
from sklearn.linear_model import SGDRegressor, ElasticNet

from lenskit import util
from .. import check
from ..matrix import CSR, sparse_ratings
from . import Predictor, ItemNeighborhood
from . import item_knn

_logger = logging.getLogger(__name__)

def _train_item(slimAlgo, item, rmat):
    # Create an ElasticNet optimization function
    opt_model = ElasticNet(alpha=slimAlgo.alpha, max_iter=slimAlgo.max_iter, tol=.00001, l1_ratio=slimAlgo.l_1_ratio, positive=True, fit_intercept=False, copy_X=False)

    # Copy the passed in matrix to avoid altering the original ratings matrix
    sp_rmat = rmat.to_scipy().copy()
    item_col = sp_rmat.getcol(item)
        
    # Zero out the column of the item before optimizing to prevent the model from optimizing for the item itself
    sp_rmat[sp_rmat[:, item].nonzero()[0], item] = 0

    opt_model.fit(sp_rmat, item_col.todense())

    # Indexes of coefficient array with positive values
    sparse_coeff_coo = opt_model.sparse_coef_.tocoo()
    return (item,  np.full(sparse_coeff_coo.nnz, item), sparse_coeff_coo.col, sparse_coeff_coo.data)

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

        binary(bool):
            Determines if the SLIM algorithm should optimize using the passed in ratings values
            or if the ratings should be transformed to binary ownership values. 
        
        nprocs(int):
            Number of threads to use when fitting the data

    Attributes:
        l_1_regularization(double): The l_1 regularization factor
        l_2_regularization(double): The l_2 regularization factor
        binary(bool): Indicates whether the rating column should be treated as ownership or rating scale
        nprocs(int): Number of threads to use when fitting the data
    """

    def __init__(self, regularization=(1.0, 2.0), binary=False, max_iter=1000, nprocs=1):
        if isinstance(regularization, tuple):
            self.regularization = regularization
            self.l_1_regularization, self.l_2_regularization = regularization
        else:
            self.regularization = regularization
            self.l_1_regularization = regularization
            self.l_2_regularization = regularization

        self.binary = binary
        self.max_iter = int(max_iter)
        self.nprocs = int(nprocs)
        
        check.check_value(self.l_1_regularization >= 0, "l_1 norm regularization value {} must be nonnegative",
                          self.l_1_regularization)
        check.check_value(self.l_2_regularization >= 0, "l_2 norm regularization {} must be nonnegative",
                          self.l_2_regularization)
        check.check_value(type(self.binary) == type(False), "Binary {} must be a boolean value",
                          self.binary)
        check.check_value(self.max_iter > 0, "Maximum number of optimization iterations {} must be a positive integer",
                          self.max_iter)
        check.check_value(self.nprocs > 0, "Number of processes {} must be a positive integer",
                          self.nprocs)


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

        if self.binary:
            data = data.copy(deep=True)
            data['rating'] = 1

        rmat, uidx, iidx = sparse_ratings(data)

        # Optimize each item independently on different threads using joblib
        item_coeff_array_tuples = Parallel(n_jobs=self.nprocs)(delayed(_train_item)(self, item, rmat) for item in range(rmat.ncols))
        _logger.info('[%s] completed calculating coefficients for %s items', self._timer, rmat.ncols)

        coeff_row = np.array([], dtype=np.int32)
        coeff_col = np.array([], dtype=np.int32)
        coeff_values = np.array([], dtype=np.float64)

        for coeff_tuple in item_coeff_array_tuples:
            # Add coefficients with proper indexes for sparse matrix
            coeff_row = np.append(coeff_row, coeff_tuple[1])
            coeff_col = np.append(coeff_col, coeff_tuple[2])
            coeff_values = np.append(coeff_values, coeff_tuple[3])

        # Create sparse coefficient matrix
        self.coefficients_ =  CSR.from_coo(coeff_row, coeff_col, coeff_values, (len(iidx), len(iidx))).to_scipy()

        self.user_index_ = uidx
        self.item_index_ = iidx
        self.ratings_matrix_ = rmat

        return self

    def predict_for_user(self, user, items=None, ratings=None):
        """
        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): NOT SUPPORTED

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        _logger.debug('Predicting %s item(s) for user %s', "ALL" if items is None else len(items), user)

        if ratings is not None:
            _logger.debug('SLIM does not support ratings fit at predict time')
            raise NotImplementedError('SLIM does not support ratings fit at predict time')
            
        upos = self.user_index_.get_loc(user)
        if -1 is upos:
            _logger.warn("The requested user was not in the original dataset")
            raise KeyError

        urow = self.ratings_matrix_.row(upos)
        urow = np.reshape(urow, (-1, 1)).transpose()

        res_series = None
        indexed_scores = {}
        if items is not None :
            
            ipos = self.item_index_.get_indexer(items)
            if -1 in ipos:
                _logger.warn("Some item ratings requested are for items missing from fit data set")

            ipos = ipos[ipos >= 0]
            _logger.debug('Scoring items: %s', items)
            raw_scores = (urow @ self.coefficients_[:, ipos])[0]

            raw_scores_index = 0
            for i in ipos:
                _logger.debug('Predicted score for %s is %s', i, raw_scores[raw_scores_index])
                indexed_scores[self.item_index_[i]] = raw_scores[raw_scores_index]
                raw_scores_index += 1
            
            res_series = pd.Series(indexed_scores, name='slim_score')

        else:
            raw_scores = (urow @ self.coefficients_)[0]
            res_series = pd.Series(raw_scores, name='slim_score', index=self.item_index_)


        return res_series

    def __str__(self):
        return 'SLIM(regularization_one={}, regularization_two={}, binary={})'.format(self.regularization_one, self.regularization_two, self.binary)

class fsSLIM(Predictor):
    """
    A user based collaborative filtering Top-N recommendation algorithm built on SLIM. Generates item coefficients faster 
    than the original SLIM algorithm by calculating coefficients for a given item using a set of ''k'' related items rather
    than the complete set of ratings/ownership.

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

        binary(bool):
            Determines if the SLIM algorithm should optimize using the passed in ratings values
            or if the ratings should be transformed to binary ownership values. 
        
        nprocs(int):
            Number of threads to use when fitting the data

        k:
            Number of related items to consider when generating coefficients, pass ''None''
            to use all related items returned by the selector algorithm

        selector:
            Predictor or Recommender Algorithm that implements ItemNeighorhoodSelector, used
            to determine the ''k'' items used in coefficient generation

    Attributes:
        l_1_regularization(double): The l_1 regularization factor
        l_2_regularization(double): The l_2 regularization factor
        binary(bool): Indicates whether the rating column should be treated as ownership or rating scale
        nprocs(int): Number of threads to use when fitting the data
        k(int):
        selector(ItemNeighborhoodSelector)
    """

    def __init__(self, regularization=(1.0, 2.0), k=100, selector=item_knn.ItemItem(None, 100, save_nbrs=100), binary=False, max_iter=1000, nprocs=1):
        if isinstance(regularization, tuple):
            self.regularization = regularization
            self.l_1_regularization, self.l_2_regularization = regularization
        else:
            self.regularization = regularization
            self.l_1_regularization = regularization
            self.l_2_regularization = regularization

        self.binary = binary
        self.k = k
        self.selector = selector
        self.max_iter = int(max_iter)
        self.nprocs = int(nprocs)
        
        check.check_value(self.l_1_regularization >= 0, "l_1 norm regularization value {} must be nonnegative",
                          self.l_1_regularization)
        check.check_value(self.l_2_regularization >= 0, "l_2 norm regularization {} must be nonnegative",
                          self.l_2_regularization)
        check.check_value(type(self.binary) == type(False), "Binary {} must be a boolean value",
                          self.binary)
        check.check_value(self.k == None or self.k > 0, "Number of related items to consider in optimization problem {} must be positive integer or None",
                          self.k )
        check.check_value(issubclass(type(self.selector), ItemNeighborhood), "Feature selector {} must be implement",
                          type(self.selector))
        check.check_value(self.max_iter > 0, "Maximum number of optimization iterations {} must be a positive integer",
                          self.max_iter)
        check.check_value(self.nprocs > 0, "Number of processes {} must be a positive integer",
                          self.nprocs)
        


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
        selector_data = data.copy(deep=True)
        self.selector.fit(selector_data)

        if self.binary:
            data = data.copy(deep=True)
            data['rating'] = 1
        
        rmat, uidx, iidx = sparse_ratings(data)

        # Optimize each item independently on different threads using joblib
        item_coeff_array_tuples = Parallel(n_jobs=self.nprocs)(delayed(_train_item)(self, item, rmat) for item in range(rmat.ncols))
        _logger.info('[%s] completed calculating coefficients for %s items', self._timer, rmat.ncols)

        coeff_row = np.array([], dtype=np.int32)
        coeff_col = np.array([], dtype=np.int32)
        coeff_values = np.array([], dtype=np.float64)

        for coeff_tuple in item_coeff_array_tuples:
            # Add coefficients with proper indexes for sparse matrix
            coeff_row = np.append(coeff_row, coeff_tuple[1])
            coeff_col = np.append(coeff_col, coeff_tuple[2])
            coeff_values = np.append(coeff_values, coeff_tuple[3])

        # Create sparse coefficient matrix
        self.coefficients_ =  CSR.from_coo(coeff_row, coeff_col, coeff_values, (len(iidx), len(iidx))).to_scipy()

        self.user_index_ = uidx
        self.item_index_ = iidx
        self.ratings_matrix_ = rmat

        return self

    def predict_for_user(self, user, items=None, ratings=None):
        """
        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): NOT SUPPORTED

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        _logger.debug('Predicting %s item(s) for user %s', "ALL" if items is None else len(items), user)

        if ratings is not None:
            _logger.debug('SLIM does not support ratings fit at predict time')
            raise NotImplementedError('SLIM does not support ratings fit at predict time')
            
        upos = self.user_index_.get_loc(user)
        if -1 is upos:
            _logger.warn("The requested user was not in the original dataset")
            raise KeyError

        urow = self.ratings_matrix_.row(upos)
        urow = np.reshape(urow, (-1, 1)).transpose()

        res_series = None
        indexed_scores = {}
        if items is not None :
            
            ipos = self.item_index_.get_indexer(items)
            if -1 in ipos:
                _logger.warn("Some item ratings requested are for items missing from fit data set")

            ipos = ipos[ipos >= 0]
            _logger.debug('Items %s have positions %s in the coefficient matrix', items, ipos )
            raw_scores = (urow @ self.coefficients_[:, ipos])[0]

            raw_scores_index = 0
            for i in ipos:
                _logger.debug('Predicted score for %s is %s', i, raw_scores[raw_scores_index])
                indexed_scores[self.item_index_[i]] = raw_scores[raw_scores_index]
                raw_scores_index += 1
            
            res_series = pd.Series(indexed_scores, name='slim_score')

        else:
            raw_scores = (urow @ self.coefficients_)[0]
            res_series = pd.Series(raw_scores, name='slim_score', index=self.item_index_)

        return res_series

    def __str__(self):
        return 'fsSLIM(regularization_one={}, regularization_two={}, k={}, selector={}, binary={})'.format(self.regularization_one, self.regularization_two, self.k, self.selector, self.binary)