#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from scipy.stats import mode


# In[4]:


X_train = pd.read_csv('TinyMNIST/trainData.csv', header = None).values
y_train = pd.read_csv('TinyMNIST/trainLabels.csv', header = None).values
X_test = pd.read_csv('TinyMNIST/testData.csv', header = None).values
y_test = pd.read_csv('TinyMNIST/testLabels.csv', header = None).values


# According to slides, a reasonable estimate for posterior probability is
# 
# $P_n(w_i|x) = \frac{k_i}{k}$
# 
# So there's no need to estimate probabilties and we can predict by majority voting in K-neighbors.

# In[24]:



def KNN_nomral_predict(X_test, X_train, k=5):
    predicts = []
    probs = []
    for i, x in enumerate(X_test):
        kn_idx = np.argsort(np.sum((X_train - x)**2, axis=1))[:k]
        probs.append(y_train[kn_idx] == y_test[i].mean())
        predicts.append(mode(y_train[kn_idx]).mode[0])
    return predicts, probs


# In[ ]:





# In[32]:


for k in [1,2,5,10,50,100,len(X_train)]:
    predicts, probs = KNN_nomral_predict(X_test, X_train, k)
    print('K:', k, end=' ')
    print('CCR:', (predicts == y_test).mean(), end=' ')
    print('Error rate:', 1 - np.mean(probs))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:





# In[ ]:





# In[ ]:




