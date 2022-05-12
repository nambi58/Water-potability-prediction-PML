#!/usr/bin/env python
# coding: utf-8

# # <center>Water Potability Prediction 💧</center>

# Name: NAMBIRAJAN R S<br>Roll No: 215229125

# ## Importing Essential Libraries, Metrics, Tools and Models

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


# ## Loading the Data

# In[2]:


df = pd.read_csv("water_potability.csv")


# ## Exploratory Data Analysis

# ***Taking a look at the first 5 rows of the dataset.***

# In[3]:


df.head()


# ***Checking the shape—i.e. size—of the data.***

# In[4]:


df.shape


# ***Learning the dtypes of columns' and how many non-null values there are in those columns.***

# In[5]:


df.info()


# ***Getting the statistical summary of dataset.***

# In[6]:


df.describe().T


# ## Handling Missing Values and Duplicates

# In[7]:


df.isna().sum()


# ***As we can see, there are missing values in columns "ph", "Sulfate" and "Trihalomethanes".***

# In[8]:


df['ph'] = df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate'] = df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))


# ***It seems that there is no duplicate in the dataset.***

# In[9]:


df.duplicated().sum()


# ## Data Visualization

# ***Visualizing the Correlation between the ten numerical real-valued variables using pairplot visualization.***

# * Blue ---> **Non Potable**
# * Orange ---> **Potable**

# In[10]:


sns.set()

sns.pairplot(df, hue="Potability")


# In[11]:


labels = ['Not Potable','Potable']
values = [df[df["Potability"]==0].shape[0], df[df["Potability"]==1].shape[0]]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, title="Potability")])
fig.show()


# ***Visualizing the linear correlations between variables using Heatmap Visualization.***

# In[12]:


plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="Greens")
plt.title("Correlations Between Variables", size=16)
plt.show()


# ## Data Preprocessing

# <h3>X, y Split</h3>

# In[13]:


X = df.drop("Potability", axis=1)
y = df["Potability"]


# <h3>Data Standardization</h3>

# In[14]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# <h3>Train-Test Split</h3>

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Machine Learning Models

# In[16]:


models = pd.DataFrame(columns=["Model", "Accuracy Score"])


# In[17]:


model_list = [("Logistic Regression", LogisticRegression(random_state=42)), 
              ("Random Forest", RandomForestClassifier(random_state=42)),
              ("Support Vector Machines", SVC(random_state=42)),
              ("Gaussian Naive Bayes", GaussianNB()),
              ("Bernoulli Naive Bayes", BernoulliNB()),
              ("KNearestNeighbor", KNeighborsClassifier(n_neighbors=2)),
              ("Decision Tree", DecisionTreeClassifier(random_state=42)),
              ("Bagging Classifier", BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=42))]


# In[18]:


for name, clf in model_list:
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    
    new_row = {"Model": name, "Accuracy Score": score}
    models = models.append(new_row, ignore_index=True)


# In[19]:


models


# ***It can be seen that the one which is yielding the most accurate result without Hyperparameter Tuning is Random Forest.***

# In[20]:


models.sort_values(by="Accuracy Score", ascending=False)


# ***Defining a ROC AUC Curve visualization function for the convenience of evaluation.***

# In[21]:


def visualize_roc_auc_curve(model, model_name):
    pred_prob = model.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:, 1], pos_label=1)
    
    score = roc_auc_score(y_test, pred_prob[:, 1])
    
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, linestyle="--", color="orange", label="ROC AUC Score: (%0.5f)" % score)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    plt.title(f"{model_name} ROC Curve", size=15)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", prop={"size": 15})
    plt.show()


# In[22]:


plt.figure(figsize=(14, 8))
sns.barplot(x=models["Model"], y=models["Accuracy Score"])
plt.title("Models' Accuracy Scores", size=15)
plt.xticks(rotation=90)
plt.show()


# <h3>Confusion Matrix and ROC Curve of Random Forest</h3>

# In[23]:


rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap="BuGn_r", fmt="d")
plt.title("Confusion Matrix of Random Forest")
plt.show()


# In[24]:


visualize_roc_auc_curve(rfc, "Random Forest")


# ### Confusion Matrix and ROC Curve of Bagging Classifier

# In[25]:


bc = BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=42)
bc.fit(X_train, y_train)
predictions = bc.predict(X_test)

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap="Purples_r", fmt="d")
plt.title("Confusion Matrix of Bagging Classifier")
plt.show()


# In[26]:


visualize_roc_auc_curve(bc, "Bagging Classifier")


# ### Confusion Matrix and ROC Curve of Decision Tree

# In[27]:


decT = DecisionTreeClassifier()
decT.fit(X_train, y_train)
predictions = decT.predict(X_test)

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap="rainbow", fmt="d")
plt.title("Confusion Matrix of Decision Tree")
plt.show()


# In[28]:


visualize_roc_auc_curve(decT, "Decision Tree")


# ## Hyperparameter Tuning

# In[29]:


tuned_models = pd.DataFrame(columns=["Model", "Accuracy Score"])


# <h3>Tuning the Random Forest</h3>

# In[30]:


param_grid_rfc = {"min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "n_estimators" :[100, 200, 500],
                  "random_state": [42]}

grid_rfc = GridSearchCV(RandomForestClassifier(), param_grid_rfc, scoring="accuracy", cv=5, verbose=0, n_jobs=-1)

grid_rfc.fit(X_train, y_train)


# In[31]:


rfc_params = grid_rfc.best_params_
rfc = RandomForestClassifier(**rfc_params)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)

new_row = {"Model": "Random Forest", "Accuracy Score": score}
tuned_models = tuned_models.append(new_row, ignore_index=True)


# <h3>Tuning the Decision Tree</h3>

# In[32]:


param_grid_dt = {'criterion':['gini','entropy'],
                  'max_depth': np.arange(3, 50)}

grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, scoring="accuracy", cv=5, verbose=0, n_jobs=-1)

grid_dt.fit(X_train, y_train)


# In[33]:


dt_params = grid_dt.best_params_
dt = DecisionTreeClassifier(**dt_params)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
score = accuracy_score(y_test, predictions)
print("Accuracy Score:", score)

new_row = {"Model": "Decision Tree Classifier", "Accuracy Score": score}
tuned_models = tuned_models.append(new_row, ignore_index=True)


# ## Model Comparison After Hyperparameter Tuning

# ***After all the Hyperparameter Tuning endeavour, there is a little bit improvement (0.016768) in the Decision Tree's result and decrease in Random Forest's accuracy scores.***

# In[34]:


tuned_models.sort_values(by="Accuracy Score", ascending=False)


# In[35]:


sns.barplot(x=tuned_models["Model"], y=tuned_models["Accuracy Score"])
plt.title("Models' Accuracy Scores After Hyperparameter Tuning")
plt.show()


# ## Conclusion

# **It can be observed that the model which yields the most accurate result is non-tuned Random Forest Classifier with an accuracy score of 0.824695. Decision tree is slightly higher accuracy after tuned.**
