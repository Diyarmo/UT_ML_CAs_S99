#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


class HMM:
    def __init__(self, states, A, B, prior):
        self.states = states
        self.A = A
        self.B = B
        self.prior = prior
    def forward(self, obs):
        fwd = [{}]     
        for y in self.states:
            fwd[0][y] = self.prior[y] * self.B[y][obs[0]]
        for t in range(1, len(obs)):
            fwd.append({})     
            for y in self.states:
                fwd[t][y] = sum((fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((fwd[len(obs) - 1][s]) for s in self.states)
        return prob


# In[2]:


states = ['H', 'S']

A = {'H': {
            'H': 0.6,
            'S': 0.4
          },
     'S': {
            'H': 0.1,
            'S': 0.9
          }
    }

B = {'H': {
            'C': 0.2,
            'S': 0.7,
            'F': 0.1
          },
     'S': {
            'C': 0.1,
            'S': 0.1,
            'F': 0.8
          }
    }

obs = "FFSCFCSCSCFF"


# In[4]:


pi = {'H': 0.9,
      'S': 0.1}
st_model = HMM(states, A, B, pi)
print("Probability in first case('H': 0.9, 'S': 0.1):", st_model.forward(obs))


# In[6]:


pi = {'H': 0.1,
      'S': 0.9}

nd_model = HMM(states, A, B, pi)
print("Probability in second case('H': 0.1, 'S': 0.9):", nd_model.forward(obs))


# این که کدام مدل صحت بیشتری دارد بستگی به جامعه ی هدف برای مدل دارد و اگر مثلا کل شهر را در نظر بگیریم که در آن تعداد مبتلایان بسیار کمتر از غیر مبتلایان است، پس احتمال پیشین حالت اول مناسب تر است ولی اگر منظور مراجعه کنندگان به بیمارستان باشد که در آن تعداد مبتلایان بسیار بیشتر است حالت دوم مناسب تر است.
# 
# به صورت کلی احتمال پیشین باید از روی جامعه ی مورد نظر تعیین شود.

# In[53]:





# In[ ]:




