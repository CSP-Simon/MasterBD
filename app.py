import matplotlib.pyplot as plt
from flask import Flask, request, render_template, session, redirect, Response, send_file
import numpy as np
import pandas as pd
import tabulate
from flask_navigation import Navigation
import io
import seaborn as sns
from utils import Table, Correlation, SamplingData


app = Flask(__name__)
nav = Navigation(app)

Diabet = pd.read_csv('diabetes.csv')
Mdata = pd.read_csv('Maternal Health Risk Data Set.csv')
Bdata = pd.read_csv('Breast Cancer.csv')

# Data cleaning
Bdata.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
Bdata.diagnosis.replace(['M', 'B'], [1, 0], inplace=True)

nav.Bar('top', [
    nav.Item('Breast Cancer data', 'b_data'),
    nav.Item('Diabetes data', 'd_data'),
    nav.Item('Maternal Health Risk Data', 'm_data'),
])


@app.route('/')
def home():
    return render_template('home.html')


# Breast Cancer routes
# ------------------------------------------------------------


@app.route('/Breast_Cancer_Data', methods=("POST", "GET"))
def b_data():
    table = Table(Bdata)
    df, df2 = table.table()

    samp = SamplingData(Bdata, "diagnosis")
    samp1 = samp.sampling_data()
    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp1




    return render_template('Bdata.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test
                           )


@app.route('/plot1')
def plot1():
    features_mean = list(Bdata.columns[1:11])
    # split dataframe into two based on diagnosis
    dfM = Bdata[Bdata['diagnosis'] == 1]
    dfB = Bdata[Bdata['diagnosis'] == 0]

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 15))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.figure
        binwidth = (max(Bdata[features_mean[idx]]) - min(Bdata[features_mean[idx]])) / 50
        ax.hist([dfM[features_mean[idx]], dfB[features_mean[idx]]], bins=np.arange(min(Bdata[features_mean[idx]]),
                                                                                   max(Bdata[features_mean[
                                                                                       idx]]) + binwidth, binwidth),
                alpha=0.5, stacked=True, label=['M', 'B'], color=['r', 'g'])
        ax.legend(loc='upper right')
        ax.set_title(features_mean[idx])
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/cor1')
def cor1():
    correlation = Correlation(Bdata)
    img = correlation.cor(19, 15)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

@app.route('/imp1')
def imp1():
    imp = SamplingData(Bdata, "diagnosis")
    img = imp.importance(15,10)
    return send_file(img, mimetype='image/png', cache_timeout=-1)






# -------------------------------------------------------------------

# Diabetes routes
# -------------------------------------------------------------------

@app.route('/Diabetes_Data', methods=("POST", "GET"))
def d_data():
    table = Table(Diabet)
    df, df2 = table.table()

    samp = SamplingData(Diabet, "Outcome")
    samp = samp.sampling_data()

    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp

    return render_template('Diabetes.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test
                           )


@app.route('/plot2')
def plot2():
    features_mean = list(Diabet.columns)
    # split dataframe into two based on diagnosis

    plt.rcParams.update({'font.size': 8})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    axes = axes.ravel()
    for col, ax in enumerate(axes):
        ax.figure
        ax.hist(Diabet[features_mean[col]])
        ax.set_title(features_mean[col])
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/cor2')
def cor2():
    correlation = Correlation(Diabet )
    img = correlation.cor(11, 9)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

@app.route('/imp2')
def imp2():
    imp = SamplingData(Diabet, "Outcome")
    img = imp.importance(13,5)
    return send_file(img, mimetype='image/png', cache_timeout=-1)
# Maternal Health Risk routes
# -----------------------------------------------------------

@app.route('/Maternal_Health_Risk_Data', methods=("POST", "GET"))
def m_data():
    table = Table(Mdata)
    df, df2 = table.table()

    samp = SamplingData(Mdata, "RiskLevel")
    samp = samp.sampling_data()
    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp

    return render_template('Mdata.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train,
                           x_test=x_test,
                           y_train=y_train,
                           y_test=y_test
                           )


@app.route('/plot3')
def plot3():
    fig, axes = plt.subplots(nrows=2, figsize=(10, 15))
    order = ['low risk', 'mid risk', 'high risk']
    sns.countplot(x='RiskLevel', data=Mdata, ax=axes[0], order=order).set(
        title="Number Of Patients In Each Risk Category")
    sns.boxplot(x='RiskLevel', y='Age', data=Mdata, ax=axes[1], order=order).set(title="Analysis In Age on Risk Level")
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)


@app.route('/cor3')
def cor3():
    correlation = Correlation(Mdata)
    img = correlation.cor(10, 9)
    return send_file(img, mimetype='image/png', cache_timeout=-1)


@app.route('/imp3')
def imp3():
    imp = SamplingData(Mdata, "RiskLevel")
    img = imp.importance(7,4)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run()
