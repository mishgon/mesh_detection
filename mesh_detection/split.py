import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def train_val_test_split(indices, n_splits=5, val_size=30, random_state=42):
    split = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(indices)
    return [[*train_test_split(train, test_size=val_size, shuffle=False), test] for train, test in split]


def stratified_train_val_test_split(indices, labels, n_splits=5, val_size=30, random_state=42):
    labels = np.asarray(labels)
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(indices, labels)
    return [[*train_test_split(train, test_size=val_size, stratify=labels[train], random_state=42), test]
            for train, test in split]
