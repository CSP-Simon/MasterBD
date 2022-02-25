import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class Table:

    def __init__(self, data):
        self.data = data

    def table(self):
        buf = io.StringIO()
        self.data.info(buf=buf)
        s = buf.getvalue()
        df = pd.DataFrame(s.split("\n"), columns=['Data info'])
        df2 = self.data.isna().sum()
        df2 = pd.DataFrame(df2, columns=['Data null values'])

        return df, df2

        return df


class Correlation:
    def __init__(self, data,):
        self.data = data


    def cor(self,x,y):
        fig, ax = plt.subplots(figsize=(x, y))
        corr = self.data.corr()
        corr.to_markdown()
        lower_triang = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
        sns.heatmap(lower_triang, vmax=.8, square=True, cmap="BuPu", annot=True)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        return img


class SamplingData:
    x_train_imp = pd.DataFrame()
    y_train_imp = pd.DataFrame()

    def __init__(self, data, target):

        self.data = data
        self.target = target

    def sampling_data(self):
        x = self.data.drop(self.target, axis=1)
        y = self.data[[self.target]]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        x_train_head = x_train.head()
        x_test_head = x_test.head()
        SamplingData.x_train_imp = SamplingData.x_train_imp.append(x_train)
        SamplingData.y_train_imp = SamplingData.y_train_imp.append(y_train)

        return x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_train_head, x_test_head

    def importance(self,x,y):
        x_train = SamplingData.x_train_imp
        y_train = SamplingData.y_train_imp
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train.values.ravel())
        fig, ax = plt.subplots(figsize=(x,y))
        sort=rf.feature_importances_.argsort()
        plt.barh(list(x_train.columns), rf.feature_importances_[sort])
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        SamplingData.x_train_imp = pd.DataFrame()
        SamplingData.y_train_imp = pd.DataFrame()
        return img
