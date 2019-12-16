from pathlib import Path
import pandas as pd
import numpy as np


class FaceDataset:
    def __init__(self, path_to_csv: Path):
        """
        :param path_to_csv: path to `training.csv` file
        """
        self.df = pd.read_csv(path_to_csv)
        self.ids = list(self.df.index)

    def load_image(self, identifier):
        return np.array(list(map(int, self.df.loc[identifier, 'Image'].split())), dtype=np.float32).reshape((96, 96))

    def load_key_points(self, identifier):
        """
        Nans mean absence of annotation
        """
        return np.flip(self.df.loc[identifier, self.df.columns[:-1]].values.astype(np.float32).reshape((-1, 2)), 1)
