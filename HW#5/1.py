#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[60]:


X_train = pd.read_csv('TinyMNIST/trainData.csv', header = None)
y_train = pd.read_csv('TinyMNIST/trainLabels.csv', header = None).values.flatten().astype(int)
X_test = pd.read_csv('TinyMNIST/testData.csv', header = None)
y_test = pd.read_csv('TinyMNIST/testLabels.csv', header = None).values.flatten().astype(int)


# In[61]:


X_train_cls = [X_train[y_train == i].values for i in range(10)]
D = X_train.shape[1]


# In[62]:


cls_n = np.array(list(map(lambda x: len(x), X_train_cls)))
priors = cls_n / np.sum(cls_n)


# In[ ]:





# In[63]:


def rect_window(u):
#     print(u)
    return (np.abs(u) <= .5).all()

def gaus_window(u):
    return (2*np.pi)**(-1/2) * np.exp(-0.5 * ((u**2).sum()))


# In[75]:


def parzen(window, hn):
    Vn_1 = 1/(hn)**D
    predicts = []
    correct_class_probs = []
    for t in range(len(X_test)):
        post_prob = []
        for i in range(10):
            u = (X_train_cls[i] - X_test.loc[t].values)/hn[i]
            phi = np.sum(list(map(window, u)))
            post_prob.append((1/cls_n[i]) * Vn_1[i] * phi)
        post_prob = np.array(post_prob)
        if post_prob.max():
            post_prob = post_prob / (post_prob.max())
        predicts.append(np.argmax(post_prob))
        correct_class_probs.append(post_prob[y_test[t]])
    return predicts, correct_class_probs
    


# In[76]:


units = [0.7, 1, 1.5]
for unit in units:
    Vn_1 = 1 / unit**D
    predicts, correct_class_probs = parzen(rect_window, np.ones(10)*unit)
    print('Unit:', unit, end=' ')
    print("Error rate:", 1 - np.mean(correct_class_probs), end=' ')
    print("CCR:", (predicts == y_test).mean())


# In[77]:


for h1 in [1, 2, 3]:
    hn = h1 / cls_n**.5
    predicts, correct_class_probs = parzen(gaus_window, hn)
    print('Unit:', h1, end=' ')
    print("Error rate:", 1 - np.mean(correct_class_probs), end=' ')
    print("CCR:", (predicts == y_test).mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




