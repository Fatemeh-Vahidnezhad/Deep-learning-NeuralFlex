import numpy as np
from sklearn.metrics import precision_score, recall_score


class Evaluation:
    def __init__(self, y, y_pred):
        self.y_pred = (y_pred > 0.6).astype(int)
        self.y = y
        # print(self.y_pred)
    
    # def error(self):
    #     return np.sum(self.y != self.y_pred)

    def accuracy(self):
        corrects = (self.y == self.y_pred).astype(int)
        # print('corrects', corrects)
        return np.mean(corrects)

    def precision(self):
        # number positive True(pt)/(positive True(pt) + positive False(pf)) (positive means prediction was 1)
        # solution 1
        # error = self.y - self.y_pred
        pt = np.sum((self.y == 1) & (self.y_pred == 1))
        pf = np.sum((self.y == 0) & (self.y_pred == 1))
        # solution 2
        # pt = np.count_nonzero(error == 0)
        # pf = np.count_nonzero(error == -1)
        if pt + pf == 0:
            # print('ttttt')
            return 0
        return round(100 * pt/(pt + pf), 3)

    def recall(self):
        # number positive True(pt)/(positive True(pt) + False Negative(fn))
        # (positive means the prediction was 1, Negative means the prediction was 0)
        # error = self.y - self.y_pred
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

    def precision_mulitclass_macro(self):

        # Converting one-hot to class labels if necessary
        y_true_labels = np.argmax(self.y, axis=1)
        y_pred_labels = np.argmax(self.y_pred, axis=1)

        # Calculate precision and recall
        precision = precision_score(y_true_labels, y_pred_labels, average='macro', , zero_division=0)
        recall = recall_score(y_true_labels, y_pred_labels, average='macro', , zero_division=0)
        return precision, recall


    # def precision_mulitclass_each(self):
    #     y_true_labels = np.argmax(self.y, axis=1)
    #     y_pred_labels = np.argmax(self.y_pred, axis=1)
    #     precision_each = precision_score(y_true_labels, y_pred_labels, average=None, zero_division=1)
    #     recall_each = recall_score(y_true_labels, y_pred_labels, average=None, zero_division=1)

    #     # print("Precision for each class:", precision_each)
    #     # print("Recall for each class:", recall_each)
    #     return precision_each, recall_each





