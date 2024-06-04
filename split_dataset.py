import pandas as pd
import numpy as np
from normalization import *


class SplitData:
    def __init__(self, path, label_name) -> None:
        self.data = pd.read_csv(path)
        self.features = len(self.data.columns)
        self.label_name = label_name

        # self.x = self.data[self.data.columns[:-1]]
        self.y = self.data[self.label_name]
        self.num_categories = len(self.y.unique())
        self.categories = self.y.unique()


    def labels(self):
        num_labels = {}
        for i in self.categories:
            num_labels[i] = self.data[self.label_name][self.data[self.label_name]==i].count()
        return num_labels

    def num_trainset(self, train_percent):
        num_train = {}
        for key, val in self.labels().items():
            num_train[key] = round(val*train_percent)
        return num_train
    
    def split_data(self, train_percent):
        self.data = self.data.sample(frac=1)
        train = []
        test = []
        for key, val in self.num_trainset(train_percent).items():
            datasubset = self.data[self.data[self.label_name] == key]
            trainSet = datasubset.iloc[:val]
            testSet = datasubset.iloc[val:]

            train.append(trainSet)
            test.append(testSet)
        df_train = pd.concat(train).reset_index(drop=True)
        df_test = pd.concat(test).reset_index(drop=True)
        return df_train, df_test


    def cross_validation(self, train, cv=2):
        pass

path = 'C:/Users/Fatemeh/Documents/Interview-Preparation/DeepLearning/data_sample.csv'
obj = SplitData(path, 'label')
# print(obj.num_trainset(0.60))
print(obj.split_data(0.7))