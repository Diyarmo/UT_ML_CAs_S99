#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_lfw_people
dataset = fetch_lfw_people(min_faces_per_person=100)
_, h, w = dataset.images.shape
data = dataset.data
labels = dataset.target
target_names = dataset.target_names


# In[6]:


data = data / 255


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(data, labels)


# In[ ]:





# In[15]:


from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB


# In[30]:


get_ipython().run_cell_magic('time', '', 'clf = GaussianNB()\nclf.fit(X_train, y_train)\nprint("Score:", clf.score(X_test, y_test))')


# In[ ]:





# In[23]:


pca = PCA(n_components=40)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[29]:


get_ipython().run_cell_magic('time', '', 'clf = GaussianNB()\nclf.fit(X_train_pca, y_train)\nprint(clf.score(X_test_pca, y_test))')


# In[ ]:




