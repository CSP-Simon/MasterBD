import matplotlib.pyplot as plt
from flask import Flask, request, render_template, session, redirect, Response, send_file
import numpy as np
import pandas as pd
import tabulate
from flask_navigation import Navigation
import io
import seaborn as sns

app = Flask(__name__)
nav = Navigation(app)
Bdata = pd.read_csv('Breast Cancer.csv')
Diabet = pd.read_csv('diabetes.csv')
Mdata = pd.read_csv('Maternal Health Risk Data Set.csv')

# Data cleaning

Bdata.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
Bdata.diagnosis.replace(['M', 'B'], [1, 0], inplace=True)

nav.Bar('top', [
    nav.Item('Breast Cancer data', 'table1'),
    nav.Item('Diabetes data', 'table2'),
    nav.Item('Maternal Health Risk Data', 'table3'),
])


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
    def __init__(self, data,x,y):
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


@app.route('/')
def home():
    return render_template('home.html')


# Breast Cancer routes
# ------------------------------------------------------------

@app.route('/table1', methods=("POST", "GET"))
def table1():
    table = Table(Bdata)
    df, df2 = table.table()

    return render_template('Bdata.html', tables=[df.to_html(classes='data', index=False), df2.to_html(classes='data')],
                           titles=df.columns.values)


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
    correlation=Correlation(Bdata,19,15)
    img=correlation.cor()
    return send_file(img, mimetype='image/png', cache_timeout=-1)


# -------------------------------------------------------------------

# Diabetes routes
# -------------------------------------------------------------------

@app.route('/table2', methods=("POST", "GET"))
def table2():
    table = Table(Diabet)
    df, df2 = table.table()

    return render_template('Diabetes.html',
                           tables=[df.to_html(classes='data', index=False), df2.to_html(classes='data')],
                           titles=df.columns.values)


@app.route('/plot2')
def plot2():
    features_mean = list(Diabet.columns)
    # split dataframe into two based on diagnosis

    plt.rcParams.update({'font.size': 8})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.figure
        ax.hist(Diabet[features_mean[idx]])
        ax.set_title(features_mean[idx])
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/cor2')
def cor2():
    correlation=Correlation(Diabet,11,9)
    img=correlation.cor()
    return send_file(img, mimetype='image/png', cache_timeout=-1)


# Maternal Health Risk routes
# -----------------------------------------------------------

@app.route('/table3', methods=("POST", "GET"))
def table3():
    table = Table(Mdata)
    df, df2 = table.table()

    return render_template('Mdata.html', tables=[df.to_html(classes='data', index=False), df2.to_html(classes='data')],
                           titles=df.columns.values)


@app.route('/plot3')
def plot3():
    fig, axes = plt.subplots(nrows=2,figsize=(10,15))
    sns.countplot(x='RiskLevel', data=Mdata,ax=axes[0]).set(title="Number Of Patients In Each Risk Category")
    sns.boxplot(x='RiskLevel', y='Age', data=Mdata, ax=axes[1]).set(title="Analysis In Age on Risk Level")
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)




@app.route('/cor3')
def cor3():
    correlation=Correlation(Mdata,10,9)
    img=correlation.cor()
    return send_file(img, mimetype='image/png', cache_timeout=-1)



# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run()
