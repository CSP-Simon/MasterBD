import matplotlib.pyplot as plt
from flask import Flask, request, render_template, session, redirect,Response,send_file
import numpy as np
import pandas as pd
import tabulate
from flask_navigation import Navigation
import io
import seaborn as sns

app = Flask(__name__)
nav = Navigation(app)
Bdata=pd.read_csv('Breast Cancer.csv')
Diabet=pd.read_csv('diabetes.csv')
Mdata=pd.read_csv('Maternal Health Risk Data Set.csv')

#Data cleaning

Bdata.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
Bdata.diagnosis.replace(['M', 'B'], [1, 0], inplace=True)


nav.Bar('top', [
    nav.Item('Breast Cancer data','table1'),
    nav.Item('Diabetes data','table2'),
    nav.Item('Maternal Health Risk Data','table3'),
])
@app.route('/')
def home():
    return render_template('home.html')

# Breast Cancer routes
# ------------------------------------------------------------

@app.route('/table1', methods=("POST", "GET"))
def table1():

    buf = io.StringIO()
    Bdata.info(buf=buf)
    s= buf.getvalue()
    df=pd.DataFrame(s.split("\n"),columns=['Breast Cancer info'])
    df2=Bdata.isna().sum()
    df2=pd.DataFrame(df2,columns=['Breast Cancer null values'])

    return render_template('Bdata.html', tables=[df.to_html(classes='data',index=False),df2.to_html(classes='data')], titles=df.columns.values)



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
        max(Bdata[features_mean[idx]]) + binwidth, binwidth),
        alpha=0.5, stacked=True, label=['M', 'B'], color=['r', 'g'])
        ax.legend(loc='upper right')
        ax.set_title(features_mean[idx])
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/cor1')
def cor1():
    fig, ax = plt.subplots(figsize=(19,15))
    corr=Bdata.corr()
    corr.to_markdown()
    lower_triang=corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    sns.heatmap(lower_triang,vmax=.8, square=True, cmap="BuPu",annot=True)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

# -------------------------------------------------------------------

# Diabetes routes
# -------------------------------------------------------------------

@app.route('/table2', methods=("POST", "GET"))
def table2():

    buf = io.StringIO()
    Diabet.info(buf=buf)
    s= buf.getvalue()
    df = pd.DataFrame(s.split("\n"), columns=['Breast Cancer info'])
    df2 = Diabet.isna().sum()
    df2 = pd.DataFrame(df2, columns=['Breast Cancer null values'])

    return render_template('Diabetes.html', tables=[df.to_html(classes='data', index=False), df2.to_html(classes='data')],
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
        ax.hist( Diabet[features_mean[idx]])
        ax.set_title(features_mean[idx])
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/cor2')
def cor2():
    fig, ax = plt.subplots(figsize=(12,10))
    corr=Diabet.corr()
    corr.to_markdown()
    lower_triang=corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    sns.heatmap(lower_triang,vmax=.8, square=True, cmap="BuPu",annot=True)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)


# Maternal Health Risk routes
#-----------------------------------------------------------

@app.route('/table3', methods=("POST", "GET"))
def table3():

    buf = io.StringIO()
    Mdata.info(buf=buf)
    s= buf.getvalue()
    df = pd.DataFrame(s.split("\n"), columns=['Breast Cancer info'])
    df2 = Mdata.isna().sum()
    df2 = pd.DataFrame(df2, columns=['Breast Cancer null values'])

    return render_template('Mdata.html', tables=[df.to_html(classes='data', index=False), df2.to_html(classes='data')],
                           titles=df.columns.values)

@app.route('/plot3')
def plot3():
    fig, ax = plt.subplots()
    sns.countplot(x = 'RiskLevel',data=Mdata)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

@app.route('/boxplot3')
def boxplot3():
    fig, ax = plt.subplots()
    sns.boxplot(x = 'RiskLevel',y='Age',data=Mdata)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

@app.route('/cor3')
def cor3():
    fig, ax = plt.subplots(figsize=(9,7))
    corr=Mdata.corr()
    corr.to_markdown()
    lower_triang=corr.where(np.tril(np.ones(corr.shape)).astype(np.bool))
    sns.heatmap(lower_triang,vmax=.8, square=True, cmap="BuPu",annot=True)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png', cache_timeout=-1)

#------------------------------------------------------------------

if __name__ == '__main__':
    app.run()