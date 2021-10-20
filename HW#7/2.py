#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report


# In[ ]:





# In[2]:


data = pd.read_csv("retailMarketing.csv", index_col=0)
X = data.drop('Age', axis=1)
y = data['Age'].replace("Young", 0).replace("Middle", 1).replace("Old", 2)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:





# In[ ]:





# In[38]:


linear_svc = SVC(kernel='linear', C=1).fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[39]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[40]:


rbf_svc = SVC(kernel='rbf', C=1).fit(X_train, y_train)
y_pred = rbf_svc.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[41]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[42]:


X_train_normal = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
X_test_normal = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))


# In[ ]:





# In[43]:


linear_svc = SVC(kernel='linear', C=1).fit(X_train_normal, y_train)
y_pred = linear_svc.predict(X_test_normal)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[44]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[45]:


rbf_svc = SVC(kernel='rbf', C=1).fit(X_train_normal, y_train)
y_pred = rbf_svc.predict(X_test_normal)
pd.DataFrame(confusion_matrix(y_test, y_pred))


# In[46]:


print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




