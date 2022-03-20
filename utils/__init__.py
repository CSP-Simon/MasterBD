import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler





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




class Correlation:
    def __init__(self, data):
        self.data = data

    def cor(self, x, y):
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
    x_test_imp = pd.DataFrame()
    y_test_imp = pd.DataFrame()

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
        SamplingData.x_test_imp = SamplingData.x_test_imp.append(x_test)
        SamplingData.y_train_imp = SamplingData.y_train_imp.append(y_train)
        SamplingData.y_test_imp = SamplingData.y_test_imp.append(y_test)

        return x_train, x_test, y_train, y_test, x_train_head, x_test_head

    def importance(self, x, y):
        x_train = SamplingData.x_train_imp
        y_train = SamplingData.y_train_imp
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train.values.ravel())
        fig, ax = plt.subplots(figsize=(x, y))
        sort = rf.feature_importances_.argsort()
        plt.barh(list(x_train.columns), rf.feature_importances_[sort])
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        SamplingData.x_train_imp = pd.DataFrame()
        SamplingData.y_train_imp = pd.DataFrame()
        return img

    def pred_models_norm(self):
        accuracy = []
        f1 = []
        model = []
        x = self.data.drop(self.target, axis=1)
        x_norm = StandardScaler().fit_transform(x)
        y = self.data[[self.target]]

        x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.25)

        lr = LogisticRegression()
        lr.fit(x_train, y_train.values.ravel())
        x_pred = lr.predict(x_test)

        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Logistic Regression')

        svc = SVC()
        svc.fit(x_train, y_train.values.ravel())
        x_pred = svc.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('SVC')


        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(x_train, y_train.values.ravel())
        x_pred = rfc.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Random Forest')


        dst = DecisionTreeClassifier(criterion='entropy')
        dst.fit(x_train, y_train.values.ravel())
        x_pred = dst.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test,x_pred, average='weighted'), 2))
        model.append('Decision Tree')



        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1})


        return output

    def pred_models(self):
        accuracy = []
        f1 = []
        model = []
        x = self.data.drop(self.target, axis=1)
        y = self.data[[self.target]]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        lr = LogisticRegression()
        lr.fit(x_train, y_train.values.ravel())
        x_pred = lr.predict(x_test)

        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Logistic Regression')

        svc = SVC()
        svc.fit(x_train, y_train.values.ravel())
        x_pred = svc.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('SVC')


        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(x_train, y_train.values.ravel())
        x_pred = rfc.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Random Forest')


        dst = DecisionTreeClassifier(criterion='entropy')
        dst.fit(x_train, y_train.values.ravel())
        x_pred = dst.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test,x_pred, average='weighted'), 2))
        model.append('Decision Tree')



        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1})


        return output

    def feat(self):
        corr = self.data.corr()
        corr.to_markdown()
        lower_triang = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
        sns.heatmap(lower_triang, vmax=.8, square=True, cmap="BuPu", annot=True)
        corr_target = abs(lower_triang[self.target])
        features = corr_target[corr_target >= 0.5]
        features = features.keys()
        features = features.delete(0)
        features = features.tolist()



        return  features