import numpy as np
from sklearn.model_selection import KFold, train_test_split


def train_val_test_split(ids, n_splits=5, val_size=30, random_state=42):
    split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(ids)
    return [[*map(np.ndarray.tolist, train_test_split(train, test_size=val_size, shuffle=False)), test.tolist()]
            for train, test in split]
