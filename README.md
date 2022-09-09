# MasterBD



<h3 align="center">Investigating Electronic Health Record </h3>


   ![Alt text](/static/home_meniu.png?raw=true )
   


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a ">About The Project</a>
      <ul>
        <li><a ">Built With</a></li>
      </ul>
    </li>
    <li><a>Implemetation</a></li>
    <ul>
        <li><a>Visualization</a></li>
        <li><a>Preparing the data</a></li>
        <li><a>Training the model</a></li>
        <li><a>Improving the model</a></li>
        <li><a>Model explainability</a></li>
      </ul>
      <li><a>Conclusions</a></li>
   
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

 This application will showcase some methods used to investigate and visualize data that will be used for creating Machine learning models for each case





### Built With

* Python
* Pycharm as IDE
* Flask
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Sklearn
* Lime
* Shap
* Html2Image

This application uses Flask for creating a web app locally 
The web application is structured very simple it only has 3 
pages that will allow the user to view each scenario based on the data set  that he will select from the home page.
The data sets represented in each page are Diabetes, Breast cancer and Maternal Health Risk
The projects for this application is structured as follow:
* A main file which contains the route for each page.  
* A template folder containing  html templates for the pages 
* A static file for the CSS and images
* And a package named utils containing  two classes with parameterized constructor for creating new object corresponding to each data set.


The data sets used in this application are Breast cancer data, Diabetes data and Maternal health Risk data






## Implementation


First step after importing the libraries is to load the data which we downloaded from Kaggle as csv an use pandas to read them


  ```sh
Diabet = pd.read_csv('diabetes.csv')
Mdata = pd.read_csv('Maternal Health Risk Data Set.csv')
Bdata = pd.read_csv('Breast Cancer.csv')
  ```
Next we create a Flask web app with items for navigation for each data which will navigate through the selected routes

  ```sh
app = Flask(__name__)
nav = Navigation(app)
nav.Bar('top', [
    nav.Item('Breast Cancer data', 'b_data'),
    nav.Item('Diabetes data', 'd_data'),
    nav.Item('Maternal Health Risk Data', 'm_data'),
])

  ```
  
   ```sh
  # ------------------------------------------------------------
# Breast Cancer routes
# ------------------------------------------------------------


@app.route('/Breast_Cancer_Data', methods=("POST", "GET"))
def b_data():
```

   ```sh
# -------------------------------------------------------------------

# Diabetes routes
# -------------------------------------------------------------------

@app.route('/Diabetes_Data', methods=("POST", "GET"))
def d_data():

  ``` 
  
 ```sh
 # -----------------------------------------------------------
# Maternal Health Risk routes
# -----------------------------------------------------------

@app.route('/Maternal_Health_Risk_Data', methods=("POST", "GET"))
def m_data():

  ``` 
  

  
  We will first create a object with the Data info class from utils package which has two methods one for creating a table displaying data information (we use that to check if there are some null values that we need to address if is the case) and also a method for creating a correlation lower triangle.
  
  We created this class for displaying information that can be applied for each data set.
  The code is the following 
  
  ```sh
class DataInfo:

    def __init__(self, data):
        self.data = data

    def table(self):
        buf = io.StringIO()
        self.data.info(buf=buf)
        s = buf.getvalue()
        df = pd.DataFrame(s.split("\n"), columns=['Data info'])

        return df

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

  ```
  We create the object in the as follow

 ```sh
data_all = DataInfo(Bdata)
df = data_all.table()
  ``` 
  
  And next as return we render the template that we created for this route and pass them 
  
   ```sh
   return render_template('Bdata.html',
                           tables=[df.to_html(classes='data',index=False)],

  ``` 
  
  In the template we will use this variable inside tags as follow
  
  ```sh
 <h1>Breast Cancer Data</h1>
<h2>Data info</h2>
{% for table in tables %}
            {{ table|safe }}
{% endfor %}

  ``` 
  
  The same applies to the rest of this application	
  
  The style used for displaying the tables can be found in the static folder as Style.css.
  
  ## Visualization
  
  After we create the object and use the first method, we get the following output
  
   ![Alt text](/static/Breast_info.png?raw=true )
   
This is the output of the Breast cancer data using the table method from Data Info class and as we can see there is a case of some null values in the unnamed column and there is an  id column.
We will remove both as there are not necessary and can negatively affect the training 

  ```sh
Bdata.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

  ``` 
  
 This is the only data that needs to be cleanned
  
  The output for the cor(correlation)  method from the same class used for Breast cancer data is as follow
  
  ![Alt text](/static/Breast_cor.png?raw=true )
  
  The correlation between features will be represented with a purple color whit a range from 0 to 8. This way we can easily spot the features correlated to the target value (diagnosis) 
  
  For more specific graphical representation we will simply create a function inside the data routes and use send_file to return the image.
  
  For the Diabetes data we will use histogram plot to make a visual representation of the distribution of the features as follow
  
  
 ```sh
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

  ``` 
  
  The output:
  
   ![Alt text](/static/Diabetes_dist.png?raw=true )
    
   By plotting the distribution of all the feature we can determine if there is a case of skewness that can be caused by outliners or if there is the existence of a normal distribution. For example, in the case of pregnancies, there is a right skewed distribution most values going for 0 same for skin thickness, insulin, Diabetes Pedigree function, and age.
   
   The distribution plot can offer useful information and with the addition of introducing two features instead of one, we can compare the result getting much more inside. In the Breast cancer data, we use this application to find the comparison between the Nucleus features vs diagnosis.
   
```sh
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
  ``` 
 The output:
  
 ![Alt text](/static/Breast_dist.png?raw=true )
 
 essentially we can observe the
target value in composition with the other features. We only use the mean nucleus
features as this offer a better understanding of the overall shape and size variables.
The first thing we see is the target value has like in the diabetes example more good
results in this case the benign( not dangerous to health)

Plotting only the target value by itself can also give information like the
rest of them. In Maternal risk data if we plot a simple bar chart of the target
value we can still have interesting observations

```sh
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
  ``` 
  
  The result shows that the actual median is higher
in the case of the high-risk feature and lower in the low and mid risk features, this
particular case indicates just like in the example before that the high risk or the bad
result tends to have higher values, plus we can observe the existence of outliners,
especially in the low and mid risk

##  Preparing the data

The
next step for preparing the data is a simple splitting in two parts training and
testing. This consists of random sampling without replacement about 70 (or a
slightly different) percent of the rows and putting them into your training set. The
remaining 30 percent is put into your test set. Before splitting the data, we also need
to separate the target value from the rest of the data and for better performance we
will remove outliners with the use of Z-score and normalize the data with a simple
formula. We will do all of the above with the following method from the Class DataMod which will be used for all data transformation. 
The class has a constructor with data and target as parameters, 
this will help with splitting the data as we need to separate the target value.

```sh
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

        return x_train, x_test, y_train, y_test,x_train_head, x_test_head
  ``` 


 The methods also returns the heads of training and testing data. We use this to check if the splitting worked.
 The output:
 
 
 ![Alt text](/static/tables_head.png?raw=true )
 
  ##  Training the model
  The multiple options for machine learning models makes it
  difficult to choose the best performing model, and for that reason we will evaluate
  the performance of four models
* Logistic regression
* Support Vector Machine
* Decision tree
* Random Forest

  For testing our models, we created a method that will use the processed data from
before as parameters and use it to test each model and the output will return a table
containing the accuracy which is a method used to calculate the accuracy of either
the faction or count of correct prediction in Python. Mathematically it represents
the ratio of the sum of true positives and true negatives out of all the predictions.
We also return the F1 score which is the harmonic mean of precision and recall,
Precision is the fraction of relevant instances among the retrieved instances, while
recall is the fraction of relevant instances that were retrieved. And the final output
in the table will be the corresponding model. The method that returns the output
is the following:

```sh
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
  ``` 
  The result of pred_models_norm on Breast cancer data:
  
   ![Alt text](/static/Breat_tr.png?raw=true )
   
   The result of pred_models_norm on Diabetes data:
   
  ![Alt text](/static/Diabetes_tr.png?raw=true )
  
  The result of pred_models_norm on Maternal Health Risk data:
  
  ![Alt text](/static/Maternal_tr.png?raw=true )
  
  The results are in, and the scores differs from model to model and to data to data. The performance is satisfying but we will want to see if we can improve our models in the next subchapter
  
  ## Improving 
  
  To improve the accuracy we will use different methods for each data.
  
  For the Breast Cancer data, we opted with a feature selection method based on how many features it has. This may cause an overfitting in general. The method that we will use is RFECV. RFECV is a feature selection method that fits a model and removes the unwanted features with cross validation
  The method is implemented in a similar way as the previous one 
  

  ```sh
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
   ``` 
   
  The method will return an output like the previous one but will also include the numbers of optimal features and will also return the transforms training data with the selected features for on the model we determined was the best.
  
  The output in comparation with the initial training for Breast cancer data
  
  ![Alt text](/static/Breat_imp.png?raw=true )
  
  The final result is a mixed one, in some models(Logistic Regression and Decision Tree) we see improvement in accuracy however in the remaining ones( SVC
  and Random Forest ) we see a decrease in accuracy. Overall, the results are still
  quite satisfactory and for this particular model we chose the Logistic regression
  model with selected features as it has a more simple data which will help with the
  processing power and time for future testing.
  
  On the other hand if we look at the Diabetes data set we can have a different
  approach as it has less features and can be possible improve by feature engineering.
  By feature engineering we simply refer to creating new feature based on the existing
  ones . In this case with a little healthcare information we can create feature for BMI,
  Insulin and Glucose. For BMI we know that there are categories for each measure’s
  range and we can have “Under”, “Healthy”,” Over”,” Obese”. For Insulin we have
  to categories Normal and Abnormal Finally for Glucose he have “low” which also
  means that we have a case of hypoglycaemia, “Normal” , “High” which is a predictor
  for prediabetes and “Very High” which is clear case of diabetes. The methods are
  as follow:

  
  ```sh
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
   ```  
   
   After we applied the methods to our data directly in the diabetes route we can display the head of the new table.
   
   ```sh
      data = Diabet.assign(BMI_RESULT=Diabet.apply(samp.set_bmi, axis=1))
      data= data.assign(INSULIN_RESULT=data.apply(samp.set_insulin, axis=1))
      Diabet_eng = data.assign(GLUCOSE_LEVEL=data.apply(samp.set_glucoz, axis=1))
      feat_eng_data = Diabet_eng.head()
   ```  
   
   The table head for the new feature data:
   
   ![Alt text](/static/Diabetes_new.png?raw=true )
   
  to get a numerical value for the new feature we will use the dummy function which will
  return binary variable that indicates whether the categorical variable takes on a
  specific value.
   ```sh
      Diabet_eng = pd.get_dummies(Diabet_eng)
   ```  
  
   
   If we run the same method used initially (pred_models_norm) with this new data and compare the result we get.
   
   ![Alt text](/static/Diabetes_imp.png?raw=true )
   
   The result is again mixes some models improve some not
   
   

  As a final method for Maternal data we improve the accuracy of the model with
  parameter tunning that uses a grid search that like RFECV offers a cross validation
  option. The method used is similar to the methods used for models prediction and
  feature selection with the main difference that we ad a grid with the parameter for
  the search for example for the SVC model the parameters are as follows:
  
  ```sh
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
                     'max_depth': [10, 20, 30],
                     'max_features': ['sqrt'],
                     'min_samples_leaf': [1],
                     'min_samples_split': [2],
                     'n_estimators': [200]}
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
   ```  
   
  
The results comparing the initial result and the new results with parameter tuning:


![Alt text](/static/Maternal_imp.png?raw=true )

The final testing scores are showing different results for each model as in previous
examples. The most remarkable improvement is in the case of the SVC model that
benefits from a 21 % increase in accuracy. On the contrary the Decision tree suffers
from the parameter tuning, this a simple indication that the default parameters are
more efficient and the chosen grid for this model was not well constructed. The
process for finding the best parameters is a complex method that requires great
processing power and time and for some models can take more parameters than
other and this example highlight this problem.

** Model explainability
  The models are built however we also want to se the explanation for the model .
There are quite a few libraries that can explain the output of the machine learning
model . For this examples we will use two of then, SHAP and Lime .

the SHAP value tells us how much each factor in the model
contributed to the forecast, for example we use SHAP for explaining the SVC model
in the Diabetes data.Figure16. The code for SHAP for SVC model is straight
forward we will need to create an object with the prediction from our model(in the case of diabetes we use the feature engineered data) and
the testing data as the following code:

 ```sh
      svc = SVC(random_state=1, kernel="linear")
    svc.fit(x_train_f, y_train_f.values.ravel())
    svc_explainer = shap.KernelExplainer(svc.predict, x_test_f)
    mpl.rcParams['savefig.directory'] = "static"
    svc_shap_values = svc_explainer.shap_values(x_test_f)
    shap.summary_plot(svc_shap_values, x_test_f)
    plt.savefig('shap_summary.png')
  ```  
 
The SHAP explainer:

![Alt text](/static/SHAP.png?raw=true )

The image above indicates the impact of the feature based on its value. If we take
a look, we can see that ”Glucose” is a top feature and if it has a high value it has a
positive SHAP value which means that than a high glucose level indicate a positive
for a diabetes result.

The SHAP method is a time consuming process and for the sake of reducing
time we use the Lime method that allows us to explain the model based on the
training result for a chosen instance
The implementation of this method will
require the values of the training values. In our case for Breast cancer we will use the
feature selected one from Random forest model and then a random instance that
will get the prediction as in the following code:


 ```sh
    rfc=RandomForestClassifier()
    rfc.fit(x_train_rfc,y_train.values.ravel())
    predict_rfc = lambda x: rfc.predict_proba(x).astype(float)
    x=x_train_rfc.values
    explainer = lime.lime_tabular.LimeTabularExplainer(x,feature_names=x_train_rfc.columns,class_names=["malignant","benign" ],kernel_width=5)
    choosen_instance =x_train_rfc.loc[[249]].values[0]
    lime_explainer = explainer.explain_instance(choosen_instance, predict_rfc, num_features=15)
    lime_explainer.save_to_file("lime.html")

    hti = Html2Image()
    hti.output_path = 'static'
    hti.screenshot(html_file='lime.html', save_as='imag_lime.png')
  ```  
Lime can save explainer as a html file and if we use Html2Image package we can get a image and displayed and use it without having to run the explainer again.

The output is the following

![Alt text](/static/lime1.png?raw=true )

The Lime method explains the model in a similar way to SHAP method . If we
take a look at the image we can se that for example if the area worst feature is low
the results indicate a malignant results.

## Conclusions

After we got the results from the models, we conclude that the process for choosing
and improving a model can be intricate and time consuming and finding the best
performing model can be very different from one data set to another. There are
certainly plenty of methods that we can use, and we only shown a few of them in
this paper however building this project we successfully demonstrate a real use of
machine learning that can help us in decision making.

  
