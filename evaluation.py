import numpy as np


class Evaluation:
    def __init__(self, y, y_pred):
        self.y_pred = np.round(y_pred)
        self.y = y

    def accuracy(self):
        corrects = (self.y == self.y_pred).astype(int)
        return np.mean(corrects)

    def precision(self):
        # number positive True(pt)/(positive True(pt) + positive False(pf)) (positive means prediction was 1)
        # solution 1
        error = self.y - self.y_pred
        pt = np.sum((self.y == 1) & (self.y_pred == 1))
        pf = np.sum((self.y == 0) & (self.y_pred == 1))
        # solution 2
        # pt = np.count_nonzero(error == 0)
        # pf = np.count_nonzero(error == -1)
        if pt + pf == 0:
            return 0
        return round(100 * pt/(pt + pf), 3)

    def recall(self):
        # number positive True(pt)/(positive True(pt) + False Negative(fn))
        # (positive means the prediction was 1, Negative means the prediction was 0)
        error = self.y - self.y_pred
        pt = np.sum((self.y == 1) & (self.y_pred == 1))
        fn = np.sum((self.y == 1) & (self.y_pred == 0))
        # solution 2: fn = np.count_nonzero(error == 1)
        if pt + fn == 0:
            return 0
        return round(100 * pt/(pt + fn), 3)

    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0
        return round(2 * ((prec * rec)/(prec + rec)), 3)



