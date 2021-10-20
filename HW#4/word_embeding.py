from hmmlearn import hmm
import numpy as np 
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

class Embed():
    def __init__(self, tr_features, n_words=200):
        self.dict = {}
        for f in tqdm(tr_features):
            for word in TextBlob(f).words:
                if str(word) in self.dict:
                    self.dict[str(word)] += 1
                else:
                    self.dict[str(word)] = 1
        self.freqs(n_words)
        
    def freqs(self , n_words):
        freq = []
        for word in self.dict:
            freq.append((word, self.dict[word]))
        freq.sort(key=lambda x:x[1] , reverse=True)

        n_words = min(n_words , len(freq))
        self.embeding = {}
        for i in range(n_words):
            self.embeding[freq[i][0]] = i
        self.embeding["BAUM"] = n_words
        return freq

    def trns_f(self, f):
        ret = []
        for word in TextBlob(f).words:
            if word in self.embeding:
                ret.append([self.embeding[word]])
            else:
                ret.append([self.embeding["BAUM"]])
        return ret
    
    def trns_df(self, tr_features):
        X = []
        lengths = []
        for f in tqdm(tr_features):
             em = self.trns_f(f)
             if len(em) == 0:
                continue
             X = X + em
             lengths.append(len(em))
        return np.array(X).reshape(-1, 1) , np.array(lengths).reshape(-1,)

# load data set
tr_data = pd.read_csv('tsa_train.csv')[:5000]
tr_f = tr_data.SentimentText

# embedding part
embed = Embed(tr_f)
embeded_sentence = embed.trns_f(tr_f[0])

# compare embeded version and the original sentence
print(tr_f[0])
print(embeded_sentence)


