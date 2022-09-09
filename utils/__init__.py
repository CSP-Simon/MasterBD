import base64
import io
from scipy import stats
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np
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
        plt.close()
        return img


class DataMod:

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def sampling_data(self):
        self.data = self.data[(np.abs(stats.zscore(self.data)) < 3).all(axis=1)]
        x = self.data.drop(self.target, axis=1)
        y = self.data[[self.target]]

        x = x - (x.mean()) / (x.std())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
        x_train_head = x_train.head()
        x_test_head = x_test.head()

        return x_train, x_test, y_train, y_test, x_train_head, x_test_head

    def importance(self, x_train, y_train, x, y):

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train.values.ravel())
        fig, axes = plt.subplots(figsize=(x, y))
        sort = rf.feature_importances_.argsort()
        plt.barh(list(x_train.columns), rf.feature_importances_[sort])
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return img

    def pred_models_norm(self, x_train, x_test, y_train, y_test):
        accuracy = []
        f1 = []
        model = []

        lr = LogisticRegression(random_state=1, max_iter=3000)
        lr.fit(x_train, y_train.values.ravel())
        x_pred = lr.predict(x_test)
        self.prefiction_lr = x_pred

        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Logistic Regression')

        svc = SVC(random_state=1, kernel="linear")
        svc.fit(x_train, y_train.values.ravel())
        x_pred = svc.predict(x_test)
        self.prefiction_svc = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('SVC')

        rfc = RandomForestClassifier(random_state=1)
        rfc.fit(x_train, y_train.values.ravel())
        x_pred = rfc.predict(x_test)
        self.prefiction_rfc = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Random Forest')

        dst = DecisionTreeClassifier(random_state=1)
        dst.fit(x_train, y_train.values.ravel())
        x_pred = dst.predict(x_test)
        self.prefiction_dst = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Decision Tree')

        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1})

        return output

    def feature_selection_cv(self, x_train, x_test, y_train, y_test):
        accuracy = []
        f1 = []
        model = []
        num_feat = []
        best_feat = []

        lr = LogisticRegression(random_state=1, max_iter=3000)
        rfecv = RFECV(estimator=lr, step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(x_train, y_train.values.ravel())
        x_pred = rfecv.predict(x_test)
        self.prefiction_lr_cv = x_pred

        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Logistic Regression')
        num_feat.append(rfecv.n_features_)
        best_feat.append(x_train.columns[rfecv.support_])

        svc = SVC(random_state=1, kernel="linear")
        rfecv = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(x_train, y_train.values.ravel())
        x_pred = rfecv.predict(x_test)
        self.prefiction_svc_cv = x_pred

        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('SVC')
        num_feat.append(rfecv.n_features_)
        best_feat.append(x_train.columns[rfecv.support_])
        rfc = RandomForestClassifier(random_state=1)
        rfecv = RFECV(estimator=rfc, step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(x_train, y_train.values.ravel())
        x_pred = rfecv.predict(x_test)
        self.prefiction_rfc_cv = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Random Forest')
        num_feat.append(rfecv.n_features_)
        best_feat.append(x_train.columns[rfecv.support_])
        selected_feature = rfecv.support_
        selected_feature = x_train[x_train.columns[selected_feature]]

        dst = DecisionTreeClassifier(random_state=1)
        rfecv = RFECV(estimator=dst, step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(x_train, y_train.values.ravel())
        x_pred = rfecv.predict(x_test)
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Decision Tree')
        num_feat.append(rfecv.n_features_)
        best_feat.append(x_train.columns[rfecv.support_])
        self.prefiction_dst_cv = x_pred

        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1,
                               'Number of optimal feature': num_feat,
                               # 'Best features': best_feat,
                               })

        return output, selected_feature

    def get_classification_report(self, y_test, prediction):
        from sklearn import metrics
        report = metrics.classification_report(y_test, prediction, output_dict=True)
        df_classification_report = pd.DataFrame(report).transpose()
        df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)

        uniqueValues, occurCount = np.unique(y_test, return_counts=True)
        frequency_actual = (occurCount[0], occurCount[1])

        uniqueValues, occurCount = np.unique(prediction, return_counts=True)
        frequency_predicted_svc = (occurCount[0], occurCount[1])

        n_groups = 2
        fig, ax = plt.subplots(figsize=(10, 5))
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8

        rects1 = plt.bar(index, frequency_actual, bar_width,
                         alpha=opacity,
                         color='g',
                         label='Actual')
        rects6 = plt.bar(index + bar_width, frequency_predicted_svc, bar_width,
                         alpha=opacity,
                         color='purple',
                         label='Predicted')
        plt.xlabel('disease Risk')
        plt.ylabel('Frequency')
        plt.title('Actual vs Predicted frequency.')
        plt.xticks(index + bar_width, ('0', '1'))
        plt.legend()
        plt.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return df_classification_report, img

    def set_bmi(self, row):
        if row["BMI"] < 18.5:
            return "Under"
        elif row["BMI"] >= 18.5 and row["BMI"] <= 24.9:
            return "Healthy"
        elif row["BMI"] >= 25 and row["BMI"] <= 29.9:
            return "Over"
        elif row["BMI"] >= 30:
            return "Obese"

    def set_insulin(self, row):
        if row["Insulin"] >= 16 and row["Insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"

    def set_glucoz(self, row):
        if row["Glucose"] < 90:
            return "Low"
        if row["Glucose"] >= 90 and row["Glucose"] <= 140:
            return "Normal"
        if row["Glucose"] >= 141 and row["Glucose"] <= 199:
            return "High"
        if row["Glucose"] >= 200:
            return "Very High"

    def par_tun(self, x_train, x_test, y_train, y_test):
        accuracy = []
        f1 = []
        model = []
        best_param = []

        lr = LogisticRegression(random_state=1, max_iter=3000)
        grid_vals = {'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1]}
        grid_lr = GridSearchCV(estimator=lr, param_grid=grid_vals, scoring='accuracy', cv=5, refit=True,
                               return_train_score=True, verbose=10)
        grid_lr.fit(x_train, y_train.values.ravel())
        x_pred = grid_lr.predict(x_test)
        self.pred_tun_lr = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Logistic Regression')
        best_param.append(grid_lr.best_params_)

        svc = SVC(random_state=1)
        grid_vals = {'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
        grid_svc = GridSearchCV(estimator=svc, param_grid=grid_vals, scoring='accuracy', cv=5, refit=True,
                                return_train_score=True, verbose=10)
        grid_svc.fit(x_train, y_train.values.ravel())
        x_pred = grid_svc.predict(x_test)
        self.pred_tun_svc = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('SVC')
        best_param.append(grid_svc.best_params_)

        rfc = RandomForestClassifier(random_state=1)
        grid_vals = {'bootstrap': [True, False],
                     'max_depth': [10, 20, 30, 40, None],
                     'max_features': ['sqrt'],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [2, 5, 10],
                     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600]}
        grid_rfc = GridSearchCV(estimator=rfc, param_grid=grid_vals, scoring='accuracy', cv=5, refit=True,
                                return_train_score=True, verbose=10)
        grid_rfc.fit(x_train, y_train.values.ravel())
        x_pred = grid_rfc.predict(x_test)
        self.pred_tun_rfc = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Random Forest')
        best_param.append(grid_rfc.best_params_)

        dst = DecisionTreeClassifier(random_state=1)
        grid_vals = {'max_depth': [2, 3, 5, 10, 20, 30, None], 'min_samples_leaf': [5, 10, 20, 50, 100],
                     'criterion': ["gini", "entropy"], 'max_features': ['sqrt', 'log2'],
                     'min_samples_split': [2, 5, 10]}
        grid_dst = GridSearchCV(estimator=dst, param_grid=grid_vals, scoring='accuracy', cv=5, refit=True,
                                return_train_score=True, verbose=10)
        grid_dst.fit(x_train, y_train.values.ravel())
        x_pred = grid_dst.predict(x_test)
        self.pred_tun_dst = x_pred
        accuracy.append(np.round(accuracy_score(y_test, x_pred), 2))
        f1.append(np.round(f1_score(y_test, x_pred, average='weighted'), 2))
        model.append('Decision Tree')
        best_param.append(grid_dst.best_params_)

        output = pd.DataFrame({'Model': model,
                               'Accuracy': accuracy,
                               'F1 score': f1,
                               'Best parameter': best_param, })
        return output
