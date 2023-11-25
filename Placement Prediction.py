#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[19]:


placement=pd.read_excel(r'/Users/BaBa/Downloads/Input Data/Prediction of Placement Status Data/Train Data.xlsx')


# In[20]:


df.head()


# In[21]:


df_copy=df.copy()


# In[22]:


df_copy.shape


# In[23]:


df_copy.dtypes


# In[24]:


df_copy.isnull().sum()


# In[26]:


df_copy.drop(['Year of Graduation','Email ID'],axis=1,inplace=True)


# In[28]:


df_copy.head()


# In[29]:


X=df_copy.iloc[:,3:6]


# In[30]:


X


# In[31]:


Y=df_copy.iloc[:,6]


# In[32]:


Y


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.30)


# In[35]:


Y.shape


# In[36]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[37]:


df_copy['Placement Status'].replace({'Placed':1,'Not Placed':0},inplace=True)


# In[38]:


df_copy.head()


# In[39]:


df_copy['Placement Status'].replace({'Not Placed':0},inplace=True)


# In[40]:


df_copy.head()


# In[41]:


df_copy['Placement Status'].replace({'Not placed':0},inplace=True)


# In[42]:


df_copy.head()


# In[43]:


df_copy.describe()


# In[44]:


import plotly.express as px  
from plotly.offline import init_notebook_mode, iplot  
init_notebook_mode(connected=True)  
  
from sklearn.decomposition import PCA  
  
from sklearn. preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  
  
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score, recall_score, f1_score  
  
import pickle  


# In[45]:


print(df_copy.duplicated().sum())  


# In[49]:


figure = px.pie(df_copy, values=df_copy['Placement Status'].value_counts().values, names=df_copy['Placement Status'].value_counts().index, title='Placed Vs Not Placed')  


# In[50]:


figure.show()


# In[51]:


print(X.shape)


# In[53]:


print(Y.shape)


# In[54]:


Y


# In[55]:


X


# In[56]:


print(X_train.shape)  
print(X_test.shape)  
print(Y_train.shape)  
print(Y_test.shape)  


# In[57]:


scaler = StandardScaler()  
X_train_scale = scaler.fit_transform(X_train)  
X_test_scale = scaler.transform(X_test)  


# In[68]:


from sklearn.ensemble import GradientBoostingClassifier


# In[69]:


gb=GradientBoostingClassifier()
gb.fit(X_train,Y_train)


# In[70]:


Y_pred=gb.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score


# In[72]:


score1=accuracy_score(Y_test,Y_pred)


# In[73]:


print(score1)


# In[80]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[81]:


lr=LogisticRegression()
lr.fit(X_train,Y_train)

svm=svm.SVC()
svm.fit(X_train,Y_train)

knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)

dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)

re=RandomForestClassifier()
re.fit(X_train,Y_train)


# In[83]:


Y_pred2=lr.predict(X_test)
Y_pred3=svm.predict(X_test)
Y_pred4=knn.predict(X_test)
Y_pred5=dt.predict(X_test)
Y_pred6=re.predict(X_test)


# In[84]:


score1=accuracy_score(Y_test,Y_pred)
score2=accuracy_score(Y_test,Y_pred2)
score3=accuracy_score(Y_test,Y_pred3)
score4=accuracy_score(Y_test,Y_pred4)
score5=accuracy_score(Y_test,Y_pred5)
score6=accuracy_score(Y_test,Y_pred6)


# In[85]:


print(score1,score2,score3,score4,score5,score6)


# In[86]:


final_data=pd.DataFrame({'Models':['GB','LR','SVM','KNN','DT','RE'],
            'ACC':[score1*100,
                   score2*100,
                   score3*100,
                   score4*100,
                   score5*100,
                   score6*100]})


# In[87]:


final_data


# In[88]:


import seaborn as sns


# In[90]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[96]:


X


# In[97]:


Y


# In[98]:


X


# In[100]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)


# In[102]:


lr=LogisticRegression()
lr.fit(X,Y_encoded)


# In[112]:


new_data.isnull().sum()


# In[115]:


new_data=pd.read_excel(r'/Users/BaBa/Downloads/Input Data/Prediction of Placement Status Data/Test Data.xlsx')


# In[116]:


new_data.head()


# In[119]:


new_data.shape()


# In[120]:


new_data.shape


# In[121]:


new_data.dtypes


# In[122]:


df_copy.dtypes


# In[124]:


new_data.head()


# In[125]:


new_data=pd.read_excel(r'/Users/BaBa/Downloads/Input Data/Prediction of Placement Status Data/Test Data.xlsx')


# In[126]:


new_data.head()


# In[127]:


new_data=pd.read_excel(r'/Users/BaBa/Downloads/Input Data/Prediction of Placement Status Data/Test Data.xlsx')


# In[128]:


new_data.head()


# In[130]:


new_data_columns = ['CGPA', 'Speaking Skills', 'ML Knowledge']
testing_df=df[new_data_columns].copy()


# In[131]:


testing_df.head()


# In[132]:


new_test_data_selected = new_data[new_data_columns].copy()


# In[133]:


predictions = lr.predict(new_test_data_selected)


# In[134]:


prob=lr.predict(new_test_data_selected)


# In[135]:


output_df = pd.DataFrame({'Predictions': predictions})


# In[136]:


output_df.to_csv('predictions.csv', index=True)


# In[137]:


output_df.head()


# In[138]:


new_data.head()


# In[139]:


import os
os.getcwd()


# In[143]:


import shutil

source_file = 'predictions.csv'

destination_directory = '/Users/BaBa/Downloads'

shutil.move(source_file, destination_directory)


# In[ ]:




