#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


# In[ ]:





# In[31]:


data = pd.read_csv("housing.data", delim_whitespace=True, header=None)
X = data.drop(13, axis=1)
y = data[13]
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[49]:


X_train_normal = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
X_test_normal = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))


# In[ ]:





# In[54]:


clf = SVR(kernel='linear', C=3).fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[55]:


print("rmse:", mean_squared_error(y_test, y_pred)**0.5)
print("mae:", mean_absolute_error(y_test, y_pred))


# In[56]:


clf = SVR(kernel='linear', C=3).fit(X_train_normal, y_train)
y_pred = clf.predict(X_test_normal)


# In[57]:


print("rmse:", mean_squared_error(y_test, y_pred)**0.5)
print("mae:", mean_absolute_error(y_test, y_pred))


# In[61]:


plt.hist(y_train, bins=20)
plt.xlabel("price")
plt.ylabel('count')
plt.show()


# In[ ]:




