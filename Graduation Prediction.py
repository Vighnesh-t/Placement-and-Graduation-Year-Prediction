#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[9]:


df = pd.read_excel("/Users/BaBa/Downloads/Input Data/Year of Graduation Data/Final Lead Data.xlsx")


# In[10]:


df['Academic Year'].fillna(0, inplace=True)
df['Current Year'].fillna(0, inplace=True)


# In[13]:


df['Academic Year'] = df['Academic Year'].astype(str)
df['Current Year'] = df['Current Year'].astype(str)


# In[14]:


df = df[df['Academic Year'].str.isnumeric() & df['Current Year'].str.isnumeric()]


# In[15]:


df['Academic Year'] = df['Academic Year'].astype(int)
df['Current Year'] = df['Current Year'].astype(int)


# In[17]:


course_duration = 4


# In[20]:


df['Year of Graduation'] = df['Current Year'] - df['Academic Year'] + course_duration


# In[21]:


df.head()


# In[22]:


output_file_path = "/Users/BaBa/Downloads/Input Data/updated_data.xlsx"


# In[23]:


df.to_excel(output_file_path, index=False)


# In[ ]:




