import numpy as np


class NormalizeData:
    def min_max_func(self, data):
        minn = np.min(data, axis=0)
        maxx = np.max(data, axis=0)
        # result between -1 and 1
        return 2 * ((data - minn) / (maxx - minn)) - 1

    def z_score(self, data):
        meann = np.mean(data, axis=0)
        stdd = np.std(data, axis=0)
        return (data - meann)/stdd

    def max_abs(self, data):
        return data/np.abs(data).max()

    def robust_scale(self, data):
        mediann = np.median(data, axis=0)
        q_25 = np.quantile(data, 0.25, axis=0)
        q_75 = np.quantile(data, 0.75, axis=0)
        IQR = q_75 - q_25
        return (data - mediann)/IQR


# import pandas as pd
# path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
# df = pd.read_csv(path)
# Nor = NormalizeData()
# print(Nor.min_max_func(df, 'col1'))
