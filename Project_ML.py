#!/usr/bin/env python
# coding: utf-8

# # Project For ML
# submitted by
# 
# Sumaiya Begum

# In[194]:


import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import linear_model



# In[195]:


columns='dept semester year total'.split()
data1=pd.read_csv("dataset2.csv")


# In[196]:


print(data1.shape)


# # here Semester: spring=0; Summer = 1;Fall = 2....
# Dept_name : BBA=0;LLb = 1;Eng=2;CSE=3...
# # year start with 2012 = 0 and end with 2019 = 8..
# 

# In[197]:


data1.head()


# In[198]:


print(data1)


# In[199]:


reg= linear_model.LinearRegression()


# In[200]:


reg.fit(data1[['dept','semester','year']],data1.total)


# In[201]:


reg.coef_


# In[202]:


reg.intercept_


# In[203]:



y_pred=reg.predict([[3,2,8]])#2 means dept:English;1 means semester: summer; and 3 means year: 2022 
print (y_pred)


# In[204]:


# 1,1,9 #dept:llb;sem:summer;year:2020
#3,0,15 #dept:cse;sem:spring;year:2025


# In[ ]:




