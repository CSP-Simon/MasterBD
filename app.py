import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file
import numpy as np
import pandas as pd

from flask_navigation import Navigation
import io
import seaborn as sns
from utils import DataInfo, DataMod

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


# ------------------------------------------------------------
# Breast Cancer routes
# ------------------------------------------------------------


@app.route('/Breast_Cancer_Data', methods=("POST", "GET"))
def b_data():
    data = DataInfo(Bdata)
    df, df2 = data.table()
    samp = DataMod(Bdata, "diagnosis")
    samp1 = samp.sampling_data()

    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp1
    pred = samp.pred_models()
    pred_norm = samp.pred_models_norm()
    features = samp.feat()
    target = ["diagnosis"]
    Bdata_f = Bdata[features + target]
    samp_f = DataMod(Bdata_f, "diagnosis")

    pred_f = samp_f.pred_models()
    pred_norm_f = samp_f.pred_models_norm()
    img_scatter = samp.plot_cluster(12, 10)

    img_cor = data.cor(19, 15)
    img_feature_imp = samp.importance(15, 10)

    return render_template('Bdata.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train.shape,
                           y_train=y_train.shape,
                           x_test=x_test.shape,
                           y_test=y_test.shape,
                           pred_output=[pred.to_html(classes='data', index=False)],
                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],
                           feat=features,
                           pred_output_f=[pred_f.to_html(classes='data', index=False)],
                           pred_output_norm_f=[pred_norm_f.to_html(classes='data', index=False)],
                           scatter_plot=img_scatter,
                           cor_plot=img_cor,
                           feature_imp=img_feature_imp,


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



# -------------------------------------------------------------------

# Diabetes routes
# -------------------------------------------------------------------

@app.route('/Diabetes_Data', methods=("POST", "GET"))
def d_data():
    data = DataInfo(Diabet)
    df, df2 = data.table()

    samp = DataMod(Diabet, "Outcome")
    samp1 = samp.sampling_data()

    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp1
    pred = samp.pred_models()
    pred_norm = samp.pred_models_norm()

    img_cor = data.cor(11, 9)
    img_feature_imp = samp.importance(16, 5)

    return render_template('Diabetes.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train.shape,
                           x_test=x_test.shape,
                           y_train=y_train.shape,
                           y_test=y_test.shape,
                           pred_output=[pred.to_html(classes='data', index=False)],
                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],

                           cor_plot=img_cor,
                           feature_imp=img_feature_imp,


                           )


@app.route('/plot2')
def plot2():
    features_mean = list(Diabet.columns)
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






# -----------------------------------------------------------
# Maternal Health Risk routes
# -----------------------------------------------------------

@app.route('/Maternal_Health_Risk_Data', methods=("POST", "GET"))
def m_data():
    data = DataInfo(Mdata)
    df, df2 = data.table()

    samp = DataMod(Mdata, "RiskLevel")
    samp1 = samp.sampling_data()
    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp1
    pred = samp.pred_models()
    pred_norm = samp.pred_models_norm()

    img_cor = data.cor(10, 9)
    img_feature_imp = samp.importance(7, 5)

    return render_template('Mdata.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],

                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],

                           x_test_head=[x_test_head.to_html(classes='data',
                                                            index=False)],
                           x_train=x_train.shape,
                           x_test=x_test.shape,
                           y_train=y_train.shape,
                           y_test=y_test.shape,
                           pred_output=[pred.to_html(classes='data', index=False)],
                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],
                           cor_plot=img_cor,
                           feature_imp=img_feature_imp,

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





# ------------------------------------------------------------------

if __name__ == '__main__':
    app.run()
