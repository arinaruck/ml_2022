# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    # 1/ N X^T (Xw - y) + 2 * lambda w = 0
    # (2N * lambda I + X^TX) w - X^Ty = 0
    # (X^TX + 2N * lambda I) w = X^Ty
    
    n, d = tx.shape
    xt = tx.T
    A = (xt @ tx + 2 * n * lambda_ * np.eye(d))
    b = xt @ y
    w = np.linalg.solve(A, b)
    return w

