from lenskit import DataWarning
import lenskit.algorithms.slim as slim
import lenskit.algorithms.item_knn as item_knn

from pathlib import Path
import logging
import os.path
import pickle

import pandas as pd
import numpy as np
from scipy import linalg as la

import pytest
from pytest import approx, mark

import lk_test_utils as lktu

_log = logging.getLogger(__name__)

ml_ratings = lktu.ml_pandas.renamed.ratings
simple_ratings = pd.DataFrame.from_records([
    (1, 6, 4.0),
    (2, 6, 2.0),
    (1, 7, 3.0),
    (2, 7, 2.0),
    (3, 7, 5.0),
    (4, 7, 2.0),
    (1, 8, 3.0),
    (2, 8, 4.0),
    (3, 8, 3.0),
    (4, 8, 2.0),
    (5, 8, 3.0),
    (6, 8, 2.0),
    (1, 9, 3.0),
    (3, 9, 4.0)
], columns=['user', 'item', 'rating'])

simple_fs_selector = item_knn.ItemItem(100, save_nbrs=3)

def test_slim_init():
    algo = slim.SLIM()

def test_slim_init_warn_negative_regularization():
    try:
        algo = slim.SLIM(-1)
    except ValueError:
        pass  # this is fine

def test_slim_train_smoke_test():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

def test_slim_train():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.user_index_, pd.Index)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0
    
    # 7 is associated with 9
    seven, nine = algo.item_index_.get_indexer([7, 9])
    _log.info('seven: %d', seven)
    _log.info('nine: %d', nine)
    assert algo.coefficients_[seven, nine] > 0

def test_slim_train_binary():
    algo = slim.SLIM(regularization=(.05, .05), binary=True)
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.user_index_, pd.Index)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0
    
    # 7 is associated with 9
    seven, nine = algo.item_index_.get_indexer([7, 9])
    _log.info('seven: %d', seven)
    _log.info('nine: %d', nine)
    assert algo.coefficients_[seven, nine] > 0

def test_slim_simple_predict():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

def test_slim_simple_predict_binary():
    algo = slim.SLIM(regularization=(.05, .05), binary=True)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

def test_slim_multiple_predict():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [6, 7])

    assert res is not None
    assert len(res) == 2
    assert 6 in res.index
    assert 7 in res.index
    assert res.index[0] == 6
    assert res.index[1] == 7
    assert not np.isnan(res.loc[7])


def test_slim_unordered_predict():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7, 6, 9])

    assert res is not None
    assert len(res) == 3
    assert 7 in res.index
    assert 6 in res.index
    assert 9 in res.index
    assert res.index[0] == 7
    assert res.index[1] == 6
    assert res.index[2] == 9
    assert not np.isnan(res.loc[7])

def test_slim_predict_all():
    algo = slim.SLIM()
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1)

    assert res is not None
    assert len(res) == 4
    assert 6 in res.index
    assert 7 in res.index
    assert 8 in res.index
    assert 9 in res.index
    assert not np.isnan(res.loc[7])

@mark.skip("Redundant with the parallel test")
def test_slim_train_big():
    algo = slim.SLIM()
    algo.fit(ml_ratings)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0

    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])
@mark.skip("Redundant with the parallel test")
def test_slim_predict_big_parallel():
    algo = slim.SLIM(nprocs=5)
    algo.fit(ml_ratings)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0


    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

    res = algo.predict_for_user(1)

    assert res is not None
    assert len(res) == len(ml_ratings.item.unique())
@mark.skip("Redundant with the parallel test")
def test_slim_predict_binary_big_parallel():
    algo = slim.SLIM(binary=True, nprocs=5)
    algo.fit(ml_ratings)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0


    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

    res = algo.predict_for_user(1)

    assert res is not None
    assert len(res) == len(ml_ratings.item.unique())

def test_fsslim_init():
    algo = slim.fsSLIM()

def test_fsslim_init_warn_negative_regularization():
    try:
        algo = slim.fsSLIM(-1)
    except ValueError:
        pass  # this is fine

def test_fsslim_init_warn_k_positive():
    try:
        algo = slim.fsSLIM(k=0)
    except ValueError:
        pass  # this is fine
    try:
        algo = slim.fsSLIM(k=-1)
    except ValueError:
        pass  # this is fine

def test_fsslim_init_warn_no_selector():
    try:
        algo = slim.fsSLIM(selector=None)
    except ValueError:
        pass  # this is fine

def test_fsslim_init_warn_invalid_selector():
    try:
        algo = slim.fsSLIM(selector=slim.SLIM())
    except ValueError:
        pass  # this is fine

def test_fsslim_train_smoke_test():
    algo = slim.fsSLIM(k=4)
    algo.fit(simple_ratings)

def test_fsslim_train():
    algo = slim.fsSLIM(k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.user_index_, pd.Index)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0
    
    # 7 is associated with 9
    seven, nine = algo.item_index_.get_indexer([7, 9])
    _log.info('seven: %d', seven)
    _log.info('nine: %d', nine)
    assert algo.coefficients_[seven, nine] > 0

def test_fsslim_train_binary():
    algo = slim.fsSLIM(regularization=(.005, .005), binary=True, k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    assert isinstance(algo.item_index_, pd.Index)
    assert isinstance(algo.user_index_, pd.Index)

    # Diagonal of the coefficient matrix is 0 and there are some values
    assert all(algo.coefficients_.diagonal() == 0)
    assert all(np.logical_not(np.isnan(algo.coefficients_.data)))
    assert len(algo.coefficients_.data) > 0
    
    # 7 is associated with 9
    seven, nine = algo.item_index_.get_indexer([7, 9])
    _log.info('seven: %d', seven)
    _log.info('nine: %d', nine)
    assert algo.coefficients_[seven, nine] > 0

def test_fsslim_simple_predict():
    algo = slim.fsSLIM(k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

def test_fsslim_simple_predict_binary():
    algo = slim.fsSLIM(regularization=(.005, .005), binary=True, k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7])

    assert res is not None
    assert len(res) == 1
    assert 7 in res.index
    assert not np.isnan(res.loc[7])

def test_fsslim_multiple_predict():
    algo = slim.fsSLIM(k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [6, 7])

    assert res is not None
    assert len(res) == 2
    assert 6 in res.index
    assert 7 in res.index
    assert res.index[0] == 6
    assert res.index[1] == 7
    assert not np.isnan(res.loc[7])


def test_fsslim_unordered_predict():
    algo = slim.fsSLIM(k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1, [7, 6, 9])

    assert res is not None
    assert len(res) == 3
    assert 7 in res.index
    assert 6 in res.index
    assert 9 in res.index
    assert res.index[0] == 7
    assert res.index[1] == 6
    assert res.index[2] == 9
    assert not np.isnan(res.loc[7])

def test_fsslim_predict_all():
    algo = slim.fsSLIM(k=3, selector=simple_fs_selector)
    algo.fit(simple_ratings)

    res = algo.predict_for_user(1)

    assert res is not None
    assert len(res) == 4
    assert 6 in res.index
    assert 7 in res.index
    assert 8 in res.index
    assert 9 in res.index
    assert not np.isnan(res.loc[7])