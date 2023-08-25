#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction using LogisticRegression

# In[1]:


#Let's start with importing necessary libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


#read the data file
data = pd.read_csv("diabetes.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


data.describe().T


# In[5]:


data.isnull().sum()


# Seems like there is no missing values in our data. Great, let's see the distribution of data:

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Let's see how data is distributed for every column
plt.figure(figsize=(15, 15), facecolor='white')
plotnumber = 1

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']  # List of colors

# Loop through each column in the dataset
for i, column in enumerate(data):
    if plotnumber <= 9:
        plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column], color=colors[i % len(colors)])  # Use modulo to cycle through colors
        plt.xlabel(column, fontsize=12)
    plotnumber += 1

plt.tight_layout()
plt.show()


# We can see there is some skewness in the data, let's deal with data.
# 
# Also, we can see there few data for columns Glucose , Insulin, skin thickenss, BMI and Blood Pressure which have value as 0. That's not possible,right? you can do a quick search to see that one cannot have 0 values for these.
# Let's deal with that. we can either remove such data or simply replace it with their respective mean values.
# Let's do the latter.

# In[7]:


#here few misconception is there lke BMI can not be zero, BP can't be zero, glucose, insuline can't be zero so lets try to fix it
# now replacing zero values with the mean of the column
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
#pregrnancies data also look skewed towards left because of some outliers, let's remove them
# q = data['Pregnancies'].quantile(0.95)
# data_cleaned = data[data['Pregnancies']<q]


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Let's see how data is distributed for every column
plt.figure(figsize=(15, 15), facecolor='white')
plotnumber = 1

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']  # List of colors

# Loop through each column in the dataset
for i, column in enumerate(data):
    if plotnumber <= 9:
        plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column], color=colors[i % len(colors)])  # Use modulo to cycle through colors
        plt.xlabel(column, fontsize=12)
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[9]:


#now we have dealt with the 0 values and data looks better. But, there still are outliers present in some columns.lets visualize it
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width= 0.5,ax=ax,  fliersize=3)


# In[10]:


#let's deal with the outliers.
q = data['Pregnancies'].quantile(0.98)
# we are removing the top 2% data from the Pregnancies column
data_cleaned = data[data['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)
# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)
# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)
# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]


# The data looks much better now than before. We will start our analysis with this data now as we don't want to loose important information.
# If our model doesn't work with accuracy, we will come back for more preprocessing.
# 
# 

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Let's see how data is distributed for every column
plt.figure(figsize=(15, 15), facecolor='white')
plotnumber = 1

colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']  # List of colors

# Loop through each column in the dataset
for i, column in enumerate(data_cleaned):
    if plotnumber <= 9:
        plt.subplot(3, 3, plotnumber)
        sns.stripplot(data[column], color=colors[i % len(colors)])  # Use modulo to cycle through colors
        plt.xlabel(column, fontsize=12)
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[20]:


#segregate the dependent and independent variable
X = data.drop(columns = ['Outcome'])
y = data['Outcome']


# In[21]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train.shape, X_test.shape


# In[22]:


import bz2,pickle
def scaler_standard(X_train, X_test):
    #scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #saving the model
    file = bz2.BZ2File('standardScalar.pkl','wb')
    pickle.dump(scaler,file)
    file.close()
    
    return X_train_scaled, X_test_scaled


# In[23]:


X_train_scaled, X_test_scaled = scaler_standard(X_train, X_test)


# This is how our data looks now after scaling. Great, now we will check for multicollinearity using VIF(Variance Inflation factor)

# In[24]:


X_train_scaled


# In[25]:


log_reg = LogisticRegression()

log_reg.fit(X_train_scaled,y_train)


# In[26]:


# r2 score
log_reg.score(X_train_scaled,y_train)


# In[27]:


# Let's use the handy function we created
def adj_r2(x,y,r2):
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[28]:


# adj_r2 score

adj_r2(X_train_scaled,y_train,log_reg.score(X_train_scaled,y_train))


# Great, our adjusted r2 score is almost same as r2 score, thus we are not being penalized for use of many features.
# 
# 
# 

# let's see how well our model performs on the test data set.

# In[29]:


y_pred = log_reg.predict(X_test_scaled)


# accuracy = accuracy_score(y_test,y_pred)
# accuracy

# In[30]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[31]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[32]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[33]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[34]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[35]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[36]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[37]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)


# In[38]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[39]:


import bz2,pickle
file = bz2.BZ2File('modelForPrediction.pkl','wb')
pickle.dump(log_reg,file)
file.close()


# In[ ]:




