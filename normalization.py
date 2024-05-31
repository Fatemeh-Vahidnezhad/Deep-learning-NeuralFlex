import numpy as np


class NormalizeData:
    def __init__(self, data):
        self.data = data

    def min_max_func(self):
        minn = np.min(self.data, axis=0)
        maxx = np.max(self.data, axis=0)
        # result between -1 and 1
        return 2 * ((self.data - minn) / (maxx - minn)) - 1

    def z_score(self):
        meann = np.mean(self.data, axis=0)
        stdd = np.std(self.data, axis=0)
        return (self.data - meann)/stdd

    def max_abs(self):
        return self.data/np.abs(self.data).max()

    def robust_scale(self):
        mediann = np.median(self.data, axis=0)
        q_25 = np.quantile(self.data, 0.25, axis=0)
        q_75 = np.quantile(self.data, 0.75, axis=0)
        IQR = q_75 - q_25
        return (self.data - mediann)/IQR


# import pandas as pd
# path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
# df = pd.read_csv(path)
# Nor = NormalizeData()
# print(Nor.min_max_func(df, 'col1'))
