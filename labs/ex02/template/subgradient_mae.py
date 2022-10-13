import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    N = y.shape[0]
    e = (y - tx @ w)
    sign = (e >= 0).reshape(-1, 1) * 2 - 1
    grad = -np.ones((1, N)) @ (tx * sign) / N
    return grad.squeeze()