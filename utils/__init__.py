import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


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
    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y

    def cor(self):
        fig, ax = plt.subplots(figsize=(self.x, self.y))
        corr = self.data.corr()
        corr.to_markdown()
        lower_triang = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
        sns.heatmap(lower_triang, vmax=.8, square=True, cmap="BuPu", annot=True)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        return img


class SamplingData:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def sampling_data(self):
        x = self.data.drop(self.target, axis=1)
        y = self.data[[self.target]]


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        x_train_head = x_train.head()
        x_test_head=x_test.head()

        return x_train.shape, x_test.shape, y_train.shape, y_test.shape,x_train_head,x_test_head
