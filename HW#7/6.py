#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


# In[4]:





# In[9]:


data = pd.read_csv("iris.data", header=None)
X = data.drop(4, axis=1)
y = data[4].replace("Iris-setosa", 0).replace("Iris-versicolor", 1).replace("Iris-virginica", 2)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:





# In[ ]:





# In[37]:


OO_clf = OneVsOneClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)
y_pred = OO_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[38]:


print(classification_report(y_test, y_pred))


# In[39]:


print(multilabel_confusion_matrix(y_test, y_pred))


# In[32]:


OR_clf = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)
y_pred = OR_clf.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[33]:


print(classification_report(y_test, y_pred))


# In[34]:


print(multilabel_confusion_matrix(y_test, y_pred))

