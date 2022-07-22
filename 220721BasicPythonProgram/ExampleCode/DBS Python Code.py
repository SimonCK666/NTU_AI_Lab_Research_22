#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("DBS_SingDollar.csv")


# In[5]:


X = df.loc[:, ["SGD"]]


# In[6]:


Y = df.loc[:, ["DBS"]]


# In[8]:


from sklearn import linear_model


# In[9]:


model = linear_model.LinearRegression()


# In[10]:


model.fit(X,Y)


# In[11]:


pred = model.predict(X)


# In[13]:


from sklearn.metrics import mean_squared_error


# In[14]:


rmse = mean_squared_error(Y, pred) ** 0.5


# In[15]:


print(rmse)


# In[16]:


print(pred)


# In[18]:


model.coef_


# In[19]:


model.intercept_


# In[ ]:




