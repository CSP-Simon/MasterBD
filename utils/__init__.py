import base64
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class DataInfo:

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

    def cor(self, x, y):
        fig, ax = plt.subplots(figsize=(x, y))
        corr = self.data.corr()
        corr.to_markdown()
        lower_triang = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
        sns.heatmap(lower_triang, vmax=.8, square=True, cmap="BuPu", annot=True)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img = base64.b64encode(img.getvalue()).decode()
        return img


class DataMod:
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
        DataMod.x_train_imp = pd.DataFrame.from_records(x_train)
        DataMod.x_test_imp = pd.DataFrame.from_records(x_test)
        DataMod.y_train_imp = pd.DataFrame.from_records(y_train)
        DataMod.y_test_imp = pd.DataFrame.from_records(y_test)

        return x_train, x_test, y_train, y_test, x_train_head, x_test_head

    def importance(self, x, y):
        x_train = DataMod.x_train_imp
        y_train = DataMod.y_train_imp
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train.values.ravel())
        fig, ax = plt.subplots(figsize=(x, y))
        sort = rf.feature_importances_.argsort()
        plt.barh(list(x_train.columns), rf.feature_importances_[sort])
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img = base64.b64encode(img.getvalue()).decode()
        DataMod.x_train_imp = pd.DataFrame()
        DataMod.y_train_imp = pd.DataFrame()
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
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Decision Tree')

        # grid_values = { 'C': np.logspace(-3,40,100)}
        # grid_lr = GridSearchCV(lr, param_grid=grid_values, scoring='recall')
        # grid_lr.fit(x_train, y_train.values.ravel())
        # x_pred = grid_lr.predict(x_test)
        #
        # accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        # f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        # model.append('Logistic regresion tuning')
        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1})


        return output

    # def pred_models(self):
    #     accuracy = []
    #     f1 = []
    #     model = []
    #     x = self.data.drop(self.target, axis=1)
    #     y = self.data[[self.target]]
    #
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    #
    #     lr = LogisticRegression()
    #     lr.fit(x_train, y_train.values.ravel())
    #     x_pred = lr.predict(x_test)
    #
    #     accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
    #     f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
    #     model.append('Logistic Regression')
    #
    #     svc = SVC()
    #     svc.fit(x_train, y_train.values.ravel())
    #     x_pred = svc.predict(x_test)
    #     accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
    #     f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
    #     model.append('SVC')
    #
    #     rfc = RandomForestClassifier(n_estimators=100)
    #     rfc.fit(x_train, y_train.values.ravel())
    #     x_pred = rfc.predict(x_test)
    #     accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
    #     f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
    #     model.append('Random Forest')
    #
    #     dst = DecisionTreeClassifier(criterion='entropy')
    #     dst.fit(x_train, y_train.values.ravel())
    #     x_pred = dst.predict(x_test)
    #     accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
    #     f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
    #     model.append('Decision Tree')
    #
    #     output = pd.DataFrame({'Model': model,
    #                            'Accuracy': accuracy,
    #                            'F1 score': f1})
    #
    #     return output

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

        return features

    def plot_cluster(self, x, y):
        fig, ax = plt.subplots(figsize=(x, y))
        x = self.data.drop(self.target, axis=1)
        x = StandardScaler().fit_transform(x)

        y = self.data[[self.target]]
        y = y.to_numpy()

        plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=20)
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img = base64.b64encode(img.getvalue()).decode()
        return img

