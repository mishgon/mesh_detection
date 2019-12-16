import numpy as np


def l2(prediction, target):
    return np.linalg.norm(prediction - target, axis=1)
