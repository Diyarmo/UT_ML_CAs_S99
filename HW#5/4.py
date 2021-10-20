#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import mode


# In[18]:


X_train = pd.read_csv('TinyMNIST/trainData.csv', header = None)
y_train = pd.read_csv('TinyMNIST/trainLabels.csv', header = None).values
X_test = pd.read_csv('TinyMNIST/testData.csv', header = None).values
y_test = pd.read_csv('TinyMNIST/testLabels.csv', header = None).values


# According to slides, a reasonable estimate for posterior probability is
# 
# $P_n(w_i|x) = \frac{k_i}{k}$
# 
# So there's no need to estimate probabilties and we can predict by majority voting in K-neighbors.

# In[ ]:





# In[15]:



def KNN_nomral_predict(X_test, X_train, k=5, d=0):
    predicts = []
    probs = []
    for i, x in enumerate(X_test):
        points = get_points_idx(kd_tree, X_train, x, d)
        kn_idx = np.argsort(np.sum((X_train[points] - x)**2, axis=1))[:k]
        probs.append(y_train[kn_idx] == y_test[i].mean())
        predicts.append(mode(y_train[kn_idx]).mode[0])
    return predicts, probs


# In[16]:


dim = 196
def make_kd_tree(points, i=0):
    if len(points) >= 10:
        while points[i].var() == 0:
            i = (i + 1) % dim
        points = points.sort_values(by = i)
        half = len(points) // 2
        new_i = (i+1)%dim
        return [
            make_kd_tree(points[: half].reset_index(drop=True), new_i),
            make_kd_tree(points[half + 1:].reset_index(drop=True), new_i),
            points[i][half],
            i,
            list(points['index'])
        ]
    else:
        return [None, None, points[i][0], i, list(points['index'])]
def get_points_idx(node, X_train, x, d):
    for d in range(d):
        if not node[0]:
            break
        i = node[3]
        pivot = node[2]
        if x[i] > pivot:
            node = node[1]
        else:
            node = node[0]
    
    return node[4]


# In[21]:


from time import time
kd_tree = make_kd_tree(X_train.reset_index().copy())
for d in range(10):
    t = time()
    predicts, probs = KNN_nomral_predict(X_test, X_train.values, 1, d)
    print('D:', d, end=' ')
    print('CCR:', (predicts == y_test).mean(), end=' ')
    print('Error rate:', 1 - np.mean(probs))
    print('time:',time() - t)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:





# In[ ]:





# In[ ]:




