#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\mishu\OneDrive\Documents\Most-Recent-Cohorts-Institution.csv(1)\diabetes group presentation.csv')
print(df)


# In[2]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 





# In[3]:


df.info()


# In[4]:


df.isnull


# In[5]:


df.shape


# In[6]:


sns.scatterplot(x = df['Age'] , y = df['Insulin'], palette = "Dark2")
plt.title("Relationships")
plt.show()


# In[7]:


corrmat = df.corr()
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Spectral_r")
plt.show()


# In[8]:


plt.figure(figsize = (15,10))

plt.subplot(2,1,1)
sns.countplot(x = 'Pregnancies', palette = 'Set2', data = df)


# In[12]:


df.describe()


# In[14]:


plt.figure(figsize = (25,20))
sns.set(color_codes = True)

plt.subplot(4,2,1)
sns.distplot(df['Glucose'], kde = False)

plt.subplot(4,2,2)
sns.distplot(df['BloodPressure'], kde = False)

plt.subplot(4,2,3)
sns.distplot(df['SkinThickness'], kde = False)

plt.subplot(4,2,4)
sns.distplot(df['Insulin'], kde = False)

plt.subplot(4,2,5)
sns.distplot(df['BMI'], kde = False)

plt.subplot(4,2,6)
sns.distplot(df['DiabetesPedigreeFunction'], kde = False)

plt.subplot(4,2,7)
sns.distplot(df['Age'], kde = False)


# In[15]:


sns.boxplot(x=df["Glucose"])


# In[16]:


sns.boxplot(x=df["BloodPressure"])


# In[17]:


sns.boxplot(x=df["SkinThickness"])


# In[18]:


sns.boxplot(x=df["BMI"])


# In[19]:


sns.boxplot(x=df["DiabetesPedigreeFunction"])


# In[20]:


sns.boxplot(x=df["Age"])


# In[21]:


sns.countplot(x = 'Pregnancies', hue = 'Outcome', palette = 'Set2', data = df)


# In[22]:


sns.catplot(x = 'Outcome', y="Glucose", kind="box", data = df)


# In[23]:


sns.catplot(x = 'Outcome', y="BloodPressure", kind="box", data = df)


# In[24]:


sns.catplot(x = 'Outcome', y="SkinThickness", kind="box", data = df)


# In[25]:


sns.catplot(x = 'Outcome', y="Insulin", kind="box", data = df)


# In[26]:


sns.catplot(x = 'Outcome', y="BMI", kind="box", data = df)


# In[27]:


sns.catplot(x = 'Outcome', y="DiabetesPedigreeFunction", kind="box", data = df)


# In[28]:


sns.catplot(x = 'Outcome', y="Age", kind="box", data = df)


# In[29]:


sns.relplot(x='BloodPressure', y = 'Glucose' , data = df)


# In[48]:


X = df.drop('Outcome', axis = 1)


# In[49]:


X = X.values


# In[50]:


y = df['Outcome']


# In[51]:


columns = df.drop('Outcome', axis = 1).columns


# In[52]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features = X
target = y

best_features = SelectKBest(score_func = chi2,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(columns),columns = ['Chi Squared Score']) 


# In[53]:


featureScores.sort_values(by = 'Chi Squared Score', ascending = False)


# In[ ]:




