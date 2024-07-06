#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Waliya451/Fraud-Detection/blob/main/Fraud_Detection_Logistic%26KNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[330]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# In[331]:


credit_card_data = pd.read_csv('./creditcard.csv')


# In[332]:


credit_card_data.head()


# In[333]:


credit_card_data.tail()


# In[334]:


credit_card_data.info()


# In[335]:


# To check teh missing values in every column
credit_card_data.isnull().sum()


# In[336]:


credit_card_data['Class'].value_counts()


# In[337]:


#dividing dataset into legit and fraud
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)
print(fraud.shape)


# In[338]:


legit.Amount.describe()


# In[339]:


fraud.Amount.describe()


# In[340]:


#compairing using mean
credit_card_data.groupby('Class').mean()


# Since the dataset is unevenly distributed so using Under sampling technique to even it out
# 

# Evenly ditributing legit and fraud into groups

# In[341]:


legit_sample = legit.sample(n=492, random_state=42)


# In[342]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset.head()


# In[343]:


new_dataset['Class'].value_counts()


# In[344]:


new_dataset.groupby('Class').mean()


# Splitting

# In[345]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[346]:


print(X)


# In[347]:


print(Y)


# Dividing the dataset into Training and Testing Data

# In[348]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=5)


# In[349]:


print(X.shape, X_train.shape, X_test.shape)


# human_bot dataset

# In[350]:


dataset = pd.read_csv("./twitter_human_bots_dataset.csv")
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.drop('created_at', axis=1, inplace=True)
dataset.drop('description', axis=1, inplace=True)
dataset.drop('lang',axis=1, inplace=True)
dataset.drop('profile_background_image_url',axis=1, inplace=True)
dataset.drop('id', axis=1, inplace=True)
dataset.drop('screen_name', axis=1, inplace=True)


# In[351]:


df = pd.DataFrame(dataset)
df


# In[352]:


# df.drop('default_profile', axis=1, inplace=True)
# df.drop('default_profile_image', axis=1, inplace=True)
# df.drop('geo_enabled', axis=1, inplace=True)


# In[353]:


pd.set_option('future.no_silent_downcasting', True)
result = df['account_type'].replace({'human': 0, 'bot': 1})
df['account_type'] = result.infer_objects(copy=False)
df.info()


# In[373]:


feats = df.drop('account_type', axis=1)
feats = feats.drop('profile_image_url', axis=1)
feats = feats.drop('location', axis=1).to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(feats, df['account_type'].to_numpy(), test_size=0.2, random_state=42)


# Training the model using Logistic Regression

# In[355]:


logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)


# Accuracy

# In[356]:


# accuracy on training data
X_train_prediction1 = logistic_model.predict(X_train)
training_data_accuracy1 = accuracy_score(X_train_prediction1, Y_train)
print('Accuracy on Training data : ', training_data_accuracy1)
# accuracy on test data
X_test_prediction1 = logistic_model.predict(X_test)
test_data_accuracy1 = accuracy_score(X_test_prediction1, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy1)


# In[357]:


print(confusion_matrix(Y_test, X_test_prediction1))
target_names = ['class 0', 'class 1']
print(classification_report(Y_test, X_test_prediction1,target_names=target_names))


# **KNN Classifier**

# In[358]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[359]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)


# In[360]:


X_train_prediction2 = knn_model.predict(X_train)
training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)


# In[361]:


X_test_prediction2 = knn_model.predict(X_test)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)


# In[362]:


print('Accuracy score on Training Data : ', training_data_accuracy2)
print('Accuracy score on Test Data : ', test_data_accuracy2)


# In[363]:


print(confusion_matrix(Y_test, X_test_prediction2))
target_names = ['class 0', 'class 1']
print(classification_report(Y_test, X_test_prediction2,target_names=target_names))


# Decision Trees

# In[364]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt


# In[365]:


decisiontree = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt_model = decisiontree.fit(X_train, Y_train)


# In[366]:


X_train_prediction3 = dt_model.predict(X_train)
training_data_accuracy3 = accuracy_score(X_train_prediction3, Y_train)


# In[367]:


X_test_prediction3 = dt_model.predict(X_test)
test_data_accuracy3 = accuracy_score(X_test_prediction3, Y_test)


# In[368]:


print('Accuracy score on Training Data : ', training_data_accuracy3)
print('Accuracy score on Test Data : ', test_data_accuracy3)


# In[369]:


print(confusion_matrix(Y_test, X_test_prediction3))
target_names = ['class 0', 'class 1']
print(classification_report(Y_test, X_test_prediction3,target_names=target_names))

