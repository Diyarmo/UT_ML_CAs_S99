#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, ClassifierMixin
import matplotlib.pyplot as plt


# In[4]:


X_train = pd.read_csv('TinyMNIST/trainData.csv', header = None)
y_train = pd.read_csv('TinyMNIST/trainLabels.csv', header = None).values.flatten()
X_test = pd.read_csv('TinyMNIST/testData.csv', header = None)
y_test = pd.read_csv('TinyMNIST/testLabels.csv', header = None).values.flatten()


# In[6]:


clf = MultinomialNB()
clf = clf.fit(X_train, y_train)
print("MultinomialNB Score:",clf.score(X_test, y_test))


# In[ ]:





# In[ ]:





# In[121]:


def NaiveS():
    results = []
    for col in X_train.columns:
        clf = MultinomialNB()
        clf = clf.fit(X_train.drop(col, axis=1), y_train)
        results.append(clf.score(X_test.drop(col, axis=1), y_test))
        
    final_results = []
    cols_history = []
    for d in range(1, 196):
        clf = MultinomialNB()
        clf = clf.fit(X_train[np.argsort(results)[:d]], y_train)
        final_results.append(clf.score(X_test[np.argsort(results)[:d]], y_test))
        cols_history.append(np.argsort(results)[:d])
    return final_results, cols_history

get_ipython().run_line_magic('time', 'final_results, cols_history = NaiveS()')


# In[122]:


best_idx = np.argmax(final_results)
print("Best Score:", final_results[best_idx])
print("Best Indexes:")
print(cols_history[best_idx])
plt.plot(final_results, '.')


# In[64]:


#B


# In[74]:





# In[112]:


def SFS():
    best_cols = []
    remaining_cols = list(X_train.columns)
    final_results = []
    cols_history = []
    for i in range(196):
        results = []
        for col in remaining_cols:
            new_cols = best_cols + [col]
            clf = MultinomialNB()
            clf = clf.fit(X_train[new_cols], y_train)
            results.append(clf.score(X_test[new_cols], y_test))
        best_idx = np.argmax(results)
        best_cols.append(remaining_cols[best_idx])
        del remaining_cols[best_idx]
        cols_history.append(best_cols)
        final_results.append(results[best_idx])
    #     print(i+1, ')',best_idx, results[best_idx])
    return final_results, cols_history
get_ipython().run_line_magic('time', 'final_results, cols_history = SFS()')


# In[113]:


best_idx = np.argmax(final_results)
print("Best Score:", final_results[best_idx])
print("Best Indexes:")
print(cols_history[best_idx])
plt.plot(final_results, '.')


# In[114]:


#C


# In[115]:


def SBE():
    best_cols = X_train.columns
    # remaining_cols = list(X_train.columns)
    final_results = []
    cols_history = []
    for i in range(195):
        results = []
        for col in best_cols:
            new_cols = best_cols.drop(col)
            clf = MultinomialNB()
            clf = clf.fit(X_train[new_cols], y_train)
            results.append(clf.score(X_test[new_cols], y_test))
        best_idx = np.argmax(results)
        best_cols = best_cols.drop(best_cols[best_idx])
        cols_history.append(best_cols)
        final_results.append(results[best_idx])
#         print(i+1, ')', best_idx, results[best_idx])
    return final_results, cols_history
get_ipython().run_line_magic('time', 'final_results, cols_history = SBE()')


# In[116]:


best_idx = np.argmax(final_results)
print("Best Score:", final_results[best_idx])
print("Best Indexes:")
print(cols_history[best_idx])
plt.plot(final_results, '.')


# In[ ]:





# In[123]:


#6


# با توجه به این نکته که در نایو سرچ یکبار متریکی برای تمام ستون‌ها به صورت جداگانه حساب می‌شود و سپس براساس آن ستون‌ها به ترتیب اسکوری که گرفته اند اضافه میشوند، پس نایو سرچ بیشترین شباهت را به پی سی ای دارد.
# 
# برای تغییر نایو سرچ به پی سی ای، باید ابتدا
# principal components 
# را حساب کنیم، و سپس متریک خود را اندازه‌ی مقادیر ویژه قرار دهیم.

# In[ ]:




