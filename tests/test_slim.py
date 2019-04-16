from lenskit import DataWarning
import lenskit.algorithms.slim as slim

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

def test_ii_init():
    algo = slim.SLIM()

def test_ii_init_warn_negative_regularization():
    try:
        algo = slim.SLIM(-1)
    except ValueError:
        pass  # this is fine

def test_ii_train_smoke_test():
    algo = slim.SLIM()
    algo.fit(ml_ratings)