import matplotlib.pyplot as plt
from flask import Flask, render_template, send_file
import numpy as np
import pandas as pd
from flask_navigation import Navigation
import io
import seaborn as sns
from utils import DataInfo, DataMod
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import base64
import shap
import lime
import lime.lime_tabular
app = Flask(__name__)
nav = Navigation(app)

Diabet = pd.read_csv('diabetes.csv')
Mdata = pd.read_csv('Maternal Health Risk Data Set.csv')
Bdata = pd.read_csv('Breast Cancer.csv')

# Data cleaning
Bdata.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

Bdata.diagnosis.replace(['M', 'B'], [1, 0], inplace=True)

Mdata.RiskLevel.replace(['low risk', 'mid risk', 'high risk'], [0, 1, 2], inplace=True)

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
    data_all = DataInfo(Bdata)
    df, df2 = data_all.table()
    features = list(Bdata.columns[1:11])



    samp = DataMod(Bdata, "diagnosis")

    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp.sampling_data()
    pred_norm = samp.pred_models_norm( x_train, x_test, y_train, y_test)


    img_cor_all = data_all.cor(19, 15)

    img_feature_imp = samp.importance(x_train,y_train,15, 10)
    feature_selection_cv,x_train_rfc=samp.feature_selection_cv( x_train, x_test, y_train, y_test)
    prediction_lr_cv=samp.prefiction_lr_cv
    clasif_report, img_report = samp.get_classification_report(y_test, prediction_lr_cv)

    lr=LogisticRegression()
    lr.fit(x_train_rfc,y_train.values.ravel())
    predict_rfc = lambda x: lr.predict_proba(x).astype(float)
    x=x_train_rfc.values
    explainer = lime.lime_tabular.LimeTabularExplainer(x,feature_names=x_train_rfc.columns,class_names=["malignant","benign" ],kernel_width=5)
    choosen_instance =x_train_rfc.loc[[249]].values[0]
    lime_explainer = explainer.explain_instance(choosen_instance, predict_rfc, num_features=15)
    lime_explainer.save_to_file("lime.html")

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

                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],
                           feat=features,


                           cor_plot_all=img_cor_all,

                           feature_selection_cv=[feature_selection_cv.to_html(classes='data', index=False)],

                           feature_imp=img_feature_imp,
                           lime_explainer=lime_explainer,
                           clasif_report=[clasif_report.to_html(classes='data', index=True)],
                           img_report=img_report,



                           )


@app.route('/plot1')
def plot1():
    features_mean = list(Bdata.columns[1:11])
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

    samp= DataMod(Diabet, "Outcome")

    img_cor = data.cor(15, 13)
    data = Diabet.assign(BMI_RESULT=Diabet.apply(samp.set_bmi, axis=1))
    data= data.assign(INSULIN_RESULT=data.apply(samp.set_insulin, axis=1))
    Diabet_eng = data.assign(GLUCOSE_LEVEL=data.apply(samp.set_glucoz, axis=1))
    feat_eng_data = Diabet_eng.head()
    fig, axes = plt.subplots(figsize=(9,8))
    sns.countplot(data=Diabet_eng, x='INSULIN_RESULT', label='Count')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_insulin= base64.b64encode(img.getvalue()).decode()
    sns.countplot(data=Diabet_eng, x='BMI_RESULT', label='Count')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_BMI = base64.b64encode(img.getvalue()).decode()
    sns.countplot(data=Diabet_eng, x='GLUCOSE_LEVEL', label='Count')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    img_glucose = base64.b64encode(img.getvalue()).decode()
    Diabet_eng = pd.get_dummies(Diabet_eng)
    Diabet_eng=Diabet_eng.drop("BMI_RESULT_Under",axis=1)
    Diabet_eng_head = Diabet_eng.head()

    samp_feat_eng=DataMod(Diabet_eng, "Outcome")

    x_train, x_test, y_train, y_test, x_train_head, x_test_head = samp.sampling_data()
    x_train_f, x_test_f, y_train_f, y_test_f, x_train_head_f, x_test_head_f = samp_feat_eng.sampling_data()

    img_feature_imp = samp.importance(x_train,y_train, 16,5)
    pred_norm= samp.pred_models_norm( x_train, x_test, y_train, y_test)
    pred_feat_eng=samp_feat_eng.pred_models_norm(x_train_f,x_test_f,y_train_f,y_test_f)
    prediction_scv_cv=samp_feat_eng.prefiction_svc
    clasif_report,img_report = samp_feat_eng.get_classification_report(y_test_f,prediction_scv_cv)
    svc = SVC(random_state=1, kernel="linear")
    svc.fit(x_train_f, y_train_f.values.ravel())
    svc_explainer = shap.KernelExplainer(svc.predict, x_test_f)
    svc_shap_values = svc_explainer.shap_values(x_test_f)
    shap_summary=shap.summary_plot(svc_shap_values, x_test_f)
    return render_template('Diabetes.html',
                           tables=[df.to_html(classes='data',
                                              index=False), df2.to_html(classes='data')],
                           x_train_head=[x_train_head.to_html(classes='data',
                                                              index=False)],
                           x_test_head=[x_test_head.to_html(classes='data',                                                          index=False)],
                           x_train=x_train.shape,
                           x_test=x_test.shape,
                           y_train=y_train.shape,
                           y_test=y_test.shape,
                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],
                           pred_feat_eng=[pred_feat_eng.to_html(classes='data', index=False)],
                           cor_plot=img_cor,
                           feature_imp=img_feature_imp,
                           feat_eng_data=[feat_eng_data.to_html(classes='data', index=False)],
                           insulin_img=img_insulin,
                           BMI_img=img_BMI,
                           glucose_img=img_glucose,
                           clasif_report=[clasif_report.to_html(classes='data', index=True)],
                           img_report=img_report,                    shap_summary=shap_summary,
                           Diabet_eng_head=[Diabet_eng_head.to_html(classes='data', index=False)] )


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
    pred_norm = samp.pred_models_norm(x_train, x_test, y_train, y_test)
    img_cor = data.cor(10, 9)
    img_feature_imp = samp.importance(x_train,y_train,7, 5)
    parameter_tun=samp.par_tun(x_train, x_test, y_train, y_test)
    pred_tun_rfc=samp.pred_tun_rfc
    clasif_report, img_report = samp.get_classification_report(y_test, pred_tun_rfc)
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
                           pred_output_norm=[pred_norm.to_html(classes='data', index=False)],
                           cor_plot=img_cor,
                           feature_imp=img_feature_imp,
                           parameter_tun=[parameter_tun.to_html(classes='data', index=False)],
                           clasif_report = [clasif_report.to_html(classes='data', index=True)],
                           img_report = img_report

                           )


@app.route('/plot3')
def plot3():
    Mdata.RiskLevel.replace([0, 1, 2], ['low risk', 'mid risk', 'high risk'], inplace=True)
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

# -------------------------------------------------------------------

# Example routes
# -------------------------------------------------------------------
