#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Analysis 

# In[1]:


# Importing data libraries
import pandas as pd
import numpy as np 
import os

# To display number rows and columns
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',None)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")


# In[2]:


Heart_attack_Capstone_project1=pd.read_csv("C:/ALEKYA/casptone/Heartattackdata.csv")


# In[3]:


Heart_attack_Capstone_project1.head(2)


# In[4]:


# finding structure of data
Heart_attack_Capstone_project1.shape


# In[5]:


print ("Shape of data: {}" . format (Heart_attack_Capstone_project1.shape))
print ("Number of rows: {}" . format (Heart_attack_Capstone_project1.shape [0]))
print ("Number of columns: {}" . format (Heart_attack_Capstone_project1.shape [1]))


# In[6]:


#Perform preliminary data inspection and report the findings as the structure of the data, missing 
# values, duplicates, etc
Heart_attack_Capstone_project1.info()


# In[7]:


Heart_attack_Capstone_project1.isnull().sum()


# In[8]:


Heart_attack_Capstone_project1.nunique()


# In[9]:


Heart_attack_Capstone_project1.duplicated().sum()


# In[10]:


# Based on the findings from the previous question, remove duplicates (if any) and treat missing 
# values using an appropriate strategy.'''  
Heart_attack_Capstone_project1.drop_duplicates(inplace=True)


# In[11]:


Heart_attack_Capstone_project1.duplicated().sum()


# In[12]:


# statistical summary of the data
Heart_attack_Capstone_project1.describe()


# In[13]:


Heart_attack_Capstone_project1.nunique().to_frame().sort_values(0)


# In[14]:


#  Identify the data variables which might be categorical in nature. Describe and explore these 
# variables using appropriate tools. For example: count plot.
display(Heart_attack_Capstone_project1['sex'].value_counts(),
       Heart_attack_Capstone_project1['sex'].value_counts(normalize=True)*100
       )


# In[15]:


for i in ['sex','cp','fbs','ca','restecg']:
    print(i,"\n\n",Heart_attack_Capstone_project1[i].value_counts(normalize=True)*100,'\n\n')
    


# In[16]:


sns.countplot(Heart_attack_Capstone_project1['sex']);


# In[17]:


#Study the occurrence of CVD across different ages.
Heart_attack_Capstone_project1.groupby([pd.cut(Heart_attack_Capstone_project1['age'],5)])['target'].mean()


# In[18]:



Heart_attack_Capstone_project1.groupby(['target'])['age'].describe()


# In[19]:


Heart_attack_Capstone_project1.sex.value_counts(normalize=True)*100


# In[20]:


Heart_attack_Capstone_project1.groupby(['sex'])['target'].describe()


# In[21]:


#Can we detect heart attack based on anomalies in resting blood pressure of the patient
Heart_attack_Capstone_project1.groupby([pd.cut(Heart_attack_Capstone_project1['trestbps'],5)])['target'].describe()


# In[22]:


Heart_attack_Capstone_project1.corr()


# In[23]:


fig = plt.figure(figsize=(30, 30))
corr_map = sns.heatmap(Heart_attack_Capstone_project1.corr(),
                      annot=True,
                      fmt='.2f',
                      cmap='coolwarm',
                      linewidth=2,
                      linecolor='green')


# In[24]:


Heart_attack_Capstone_project1.corr()['target']


# In[25]:


#can we detect a heart attack based on anamalies
Heart_attack_Capstone_project1.groupby([pd.cut(Heart_attack_Capstone_project1['chol'],5)])['target'].describe()


# In[26]:


#what can be concluded about the relationship between peak exercise and the occuerence of heart attack
Heart_attack_Capstone_project1.groupby(['slope'])['target'].mean()


# In[27]:


# Is thalassemia a major cause of cvd?
Heart_attack_Capstone_project1.groupby([pd.cut(Heart_attack_Capstone_project1['thalach'],5)])['target'].mean()


# In[28]:


Heart_attack_Capstone_project1.head()


# In[29]:


# How are the other factors determing the occurance of cvd?
sns.pairplot(Heart_attack_Capstone_project1)


# # Perform Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[31]:


Heart_attack_Capstone_project1.columns


# In[32]:


x=Heart_attack_Capstone_project1.drop('target',axis=1) #Features
y=Heart_attack_Capstone_project1.target #Label


# In[33]:


x.head(2)


# In[34]:


y.head(2)


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# ### Sklearn Logistic

# In[36]:


model=LogisticRegression()

model.fit(x_train,y_train)


# In[37]:


model.score(x_test,y_test)


# In[38]:


y_pred=model.predict(x_test)
y_pred


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report


# In[40]:


confusion_matrix(y_test,y_pred)


# In[41]:


print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




