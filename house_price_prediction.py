#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("D:\B TECH\project\house price\data.csv")
df


# In[5]:


df.isna().sum()


# In[6]:


df.head()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.heatmap(df.corr(),annot=True)


# In[9]:


df.columns


# In[10]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
Y=df['Price']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.40)


# In[13]:


x_train


# In[14]:


y_test


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


lg=LinearRegression()


# In[17]:


lg.fit(x_train,y_train)


# In[18]:


predict=lg.predict(x_test)


# In[19]:


plt.scatter(y_test,predict)


# In[20]:


sns.distplot((y_test-predict),bins=50)


# In[24]:


score=lg.score(x_test,predict)
print("accuracy of an model is : ",score)


# In[ ]:





# In[ ]:




