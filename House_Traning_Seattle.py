#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('../DATA/kc_house_data.csv')


# In[5]:


#df.isnull().sum()


# In[6]:


df.describe().transpose()


# In[12]:


plt.figure(figsize=(10,6))
sns.distplot(df['price'])


# In[13]:


sns.countplot(df['bedrooms'])


# In[20]:


df.corr()['price'].sort_values()


# In[22]:


sns.scatterplot(x='price', y='sqft_living', data=df)


# In[24]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df, hue='price')


# In[25]:


df.sort_values('price', ascending=False).head(20)


# In[26]:


non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]


# In[32]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,
                edgecolor=None, alpha=0.2,palette= 'RdYlGn',hue='price')


# In[33]:


df.head()


# In[34]:


df= df.drop('id', axis=1)


# In[35]:


df['date'] = pd.to_datetime(df['date'])


# In[36]:


df['date']


# In[38]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[39]:


df.head


# In[40]:


df.head()


# In[43]:


plt.figure(figsize=(10,6))
sns.boxplot(x='month', y='price', data=df)


# In[46]:


df.groupby('year').mean()['price'].plot()


# In[47]:


df = df.drop('date', axis=1)


# In[48]:


df.columns


# In[50]:


#df['zipcode'].value_counts()


# In[51]:


df = df.drop('zipcode',axis=1)


# In[52]:


X = df.drop('price',axis=1).values
y= df['price'].values


# In[53]:


X


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[56]:


from sklearn.preprocessing import MinMaxScaler


# In[57]:


scaler = MinMaxScaler()


# In[58]:


X_train = scaler.fit_transform(X_train)


# In[60]:


X_test = scaler.transform(X_test)


# In[61]:


import tensorflow as tf


# In[62]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation


# In[64]:


model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimzer='adam', loss='mse')


# In[65]:


model.fit(x=X_train, y=y_train, 
          validation_data=(X_test,y_test), 
         batch_size=128, epochs=400)


# In[68]:


losses = pd.DataFrame(model.history.history)


# In[69]:


losses.plot()


# In[70]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[71]:


predictions = model.predict(X_test)


# In[73]:


mean_squared_error(y_test, predictions)**0.5


# In[74]:


mean_absolute_error(y_test, predictions)


# In[77]:


df['price'].describescribe()


# In[78]:


explained_variance_score(y_test, predictions)


# In[81]:


plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
plt.plot(y_test,y_test,'r')


# Decent score for prediction considering its hitting the majority of houses

# Maybe worth, training on the bottom 99%

# In[90]:


single_house = df.drop('price', axis=1).iloc[0]


# In[91]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# Dropping first house in the DF to test model to see how well it would perform if house 1 was a new house

# In[93]:


model.predict(single_house)


# In[94]:


df.head()


# Over shooting

# Could retrain to drop top 1-2% of the most expensive houses, not too far off but could be retrained

# In[ ]:




