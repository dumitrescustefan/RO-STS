import  numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

class Pearson():

    def __init__(self):
        self.outputs = []
        self.targets = []

    def update(self, y, y_hat):
        self.outputs.append(y)
        self.targets.append(y_hat)

    def get_score(self):
        y = np.concatenate(self.outputs)
        y_hat = np.concatenate(self.targets)
        cost = pearsonr(y, y_hat)
        self.outputs = []
        self.targets = []

        return cost


class Spearman():

    def __init__(self):
        self.outputs = []
        self.targets = []

    def update(self, y, y_hat):
        self.outputs.append(y)
        self.targets.append(y_hat)

    def get_score(self):
        y = np.concatenate(self.outputs)
        y_hat = np.concatenate(self.targets)
        cost = spearmanr(y, y_hat)
        self.outputs = []
        self.targets = []

        return cost