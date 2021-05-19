#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[77]:


import pickle


# In[78]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[80]:


Home = pd.read_csv('Residency.csv')
Home.head()


# In[82]:


Home.drop('Area live-in Satisfaction', axis=1, inplace=True)
Home.drop('Stress Management', axis=1, inplace=True)


# In[83]:


Home.Residence.unique()


# In[84]:


#Home['Nearby Places'].value_counts()


# In[85]:


# Home['Residence'].value_counts()


# In[86]:


Home['Residence'].value_counts()


# In[87]:


Home.Residence.hist(bins=30, alpha=0.5)
plt.show()


# In[88]:


Home.isna().sum()


# In[89]:


Home.info()


# In[90]:


# Home.head()


# In[91]:


# Home.head()


# # Categorical to Numeric

# In[92]:


labelencoder = LabelEncoder()
Home['Nearby Places'] = labelencoder.fit_transform(Home['Nearby Places'])
Home['Area related Info'] = labelencoder.fit_transform(
    Home['Area related Info'])
Home['Nature'] = labelencoder.fit_transform(Home['Nature'])
Home['MentalPeace'] = labelencoder.fit_transform(Home['MentalPeace'])
Home['Reaction on lack of something'] = labelencoder.fit_transform(
    Home['Reaction on lack of something'])
Home['Free time activities'] = labelencoder.fit_transform(
    Home['Free time activities'])
Home['GoOut'] = labelencoder.fit_transform(Home['GoOut'])
#Home['Stress Management'] = labelencoder.fit_transform(Home['Stress Management'])
Home['Descrimination'] = labelencoder.fit_transform(Home['Descrimination'])
Home['Outing Preference'] = labelencoder.fit_transform(
    Home['Outing Preference'])
Home['Residence'] = labelencoder.fit_transform(Home['Residence'])


# In[93]:


Home['Residence'].value_counts()


# In[94]:


corrmat = Home.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(Home[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[95]:


X = Home.drop('Residence', axis=1).values
y = Home['Residence'].values


# In[96]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# In[97]:


X_train[0]


# In[98]:


# try some regression
random_forest = RandomForestClassifier(n_estimators=10)
rf = random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


# In[99]:


acc_random_forest


# In[100]:


Y_prediction


# In[101]:


X_test[0]


# In[102]:


Y_prediction[0]


# In[103]:


pickle.dump(rf, open('iri.pkl', 'wb'))


# In[104]:


Y_prediction = random_forest.predict([[5, 1, 2, 2, 1, 0, 2, 1, 1, 0]])


# In[105]:


Y_prediction[0]


# In[ ]:


# %%
