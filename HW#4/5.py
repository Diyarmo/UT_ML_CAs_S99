#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import nltk
import string
import gensim
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

from hmmlearn.hmm import  GaussianHMM, GMMHMM

stop_words = nltk.corpus.stopwords.words('english')


# In[2]:


train_data = pd.read_csv('tsa_train.csv')[['Sentiment', 'SentimentText']]
train_data, valid_data = train_test_split(train_data, test_size=0.2)


# In[ ]:





# In[3]:


lemmatizer = nltk.WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:                    
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)

    res_words = []
    for word, tag in wn_tagged:
        if tag is None:                        
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(res_words)


# In[4]:


def sent_sep(data):
    corpus = []
    for i in range(2):
        cat_data = data[data.Sentiment == i].SentimentText
        corpus.append(cat_data)
    return corpus

train_corpus = sent_sep(train_data)
valid_corpus = sent_sep(valid_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "

def transform(X):
    X = X.str.replace(r"@[a-zA-Z0-9_]* ", " ")
    
    # Keeping only the word after the #    
    X = X.str.replace("#", "")
    
    X = X.str.replace(r"[-\.\n]", "")
    
    # Removing HTML garbag
    X = X.str.replace(r"&\w+;", "")
    
    # Removing links
    X = X.str.replace(r"https?://\S*", "")
    
    
    # replace repeated letters with only two occurences
    # heeeelllloooo => heelloo
    X = X.str.replace(r"(.)\1+", r"\1\1")
    
    # mark emoticons as happy or sad
    X = X.str.replace(HAPPY_EMO, " happyemoticons ")
    X = X.str.replace(SAD_EMO, " sademoticons ")
    
    X = X.str.lower()
    X = X.str.translate(str.maketrans('', '', string.punctuation))
    X = X.str.translate(str.maketrans('', '', '1234567890'))

    return X


# In[6]:


def clean_corpus(corpus):
    new_corpus = []
    for c in corpus:
        new_c = transform(c)
        new_c = new_c.apply(lemmatize_sentence)
        new_c = new_c.apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
        new_corpus.append(new_c)
    return new_corpus


# In[10]:


train_clean = clean_corpus(train_corpus)
valid_clean = clean_corpus(valid_corpus)


# In[11]:


def join_corpus(corpus):
    joined_corpus = []
    for c in corpus:
        joined_corpus.append(" ".join(c.values))
    return joined_corpus
train_joined = join_corpus(train_clean)
valid_joined = join_corpus(valid_clean)


# In[12]:


X_train_len = []
X_valid_len = []
for i in range(2):
    X_train_len.append(np.array(list(map(lambda x: len(x.split()), train_clean[i]))))
    X_valid_len.append(np.array(list(map(lambda x: len(x.split()), valid_clean[i]))))


# In[13]:


documents = [_text.split() for _text in train_clean[0]]
documents += [_text.split() for _text in train_clean[1]]


# In[14]:


import gensim
w2v_model = gensim.models.word2vec.Word2Vec(size=50, 
                                            window=5, 
                                            min_count=8, 
                                            workers=8)


# In[15]:


w2v_model.build_vocab(documents)


# In[16]:


w2v_model.train(documents, total_examples=len(documents), epochs=16)


# In[17]:


wv = w2v_model.wv


# In[18]:


X_train = []
X_valid = []
for i in range(2):
    X_train.append(list(map(lambda x: [ list(wv[w]) for w in x.split() if w in wv], train_clean[i].values)))
    X_valid.append(list(map(lambda x: [ list(wv[w]) for w in x.split() if w in wv], valid_clean[i].values)))


# In[19]:


for i in range(2):
    X_train[i] = [x for x in X_train[i] if x != []]
    X_valid[i] = [x for x in X_valid[i] if x != []]


# In[20]:


X_train_len = []
X_valid_len = []
for i in range(2):
    X_train_len.append(np.array(list(map(lambda x: len(x), X_train[i]))))
    X_valid_len.append(np.array(list(map(lambda x: len(x), X_valid[i]))))


# In[21]:


X_train_flatten = []
X_valid_flatten = []
for i in range(2):
    flatten = []
    for l in X_train[i]:
        flatten.extend(l)
    X_train_flatten.append(np.array(flatten))
    
    flatten = []
    for l in X_valid[i]:
        flatten.extend(l)
    X_valid_flatten.append(np.array(flatten))


# In[22]:


model0 = GaussianHMM(covariance_type='full')
model1 = GaussianHMM(covariance_type='full')

model0 = model0.fit(X_train_flatten[0], X_train_len[0])
model1 = model1.fit(X_train_flatten[1], X_train_len[1])


# In[23]:


y_true = np.array([0]*len(X_valid_len[0]) + [1]*len(X_valid_len[1]))
predicts = []

for c in range(2):
    idx = X_valid_len[c].cumsum()
    for i in range(len(idx)):
        if i==0:
            x = X_valid_flatten[c][:idx[i]]
        else:
            x = X_valid_flatten[c][idx[i-1]:idx[i]]
        predicts.append(np.argmax([model0.score(x), model1.score(x)]))
    


# In[24]:


from sklearn.metrics import confusion_matrix, recall_score, precision_score


# In[25]:


print("GaussianHMM:")
print('confusion matrix:')
print(confusion_matrix(y_true=y_true, y_pred= predicts))
print('precision:\t', precision_score(y_true=y_true, y_pred= predicts))
print('recall:\t', recall_score(y_true=y_true, y_pred= predicts))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[476]:





# In[ ]:





# In[551]:





# In[585]:





# In[584]:


predicts.mean()


# In[586]:


predicts.mean()


# In[ ]:





# In[565]:


X_valid_len[0].sum()


# In[ ]:




