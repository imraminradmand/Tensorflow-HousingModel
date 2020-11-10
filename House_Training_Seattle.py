#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('../DATA/kc_house_data.csv')


# In[3]:


#df.isnull().sum()


# In[4]:


df.describe().transpose()


# In[5]:


plt.figure(figsize=(10,6))
sns.distplot(df['price'])


# In[6]:


sns.countplot(df['bedrooms'])


# In[7]:


df.corr()['price'].sort_values()


# In[8]:


sns.scatterplot(x='price', y='sqft_living', data=df)


# In[9]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df, hue='price')


# In[10]:


df.sort_values('price', ascending=False).head(20)


# In[11]:


non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]


# In[12]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,
                edgecolor=None, alpha=0.2,palette= 'RdYlGn',hue='price')


# In[13]:


df.head()


# In[14]:


df= df.drop('id', axis=1)


# In[15]:


df['date'] = pd.to_datetime(df['date'])


# In[16]:


df['date']


# In[17]:


df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)


# In[18]:


df.head


# In[19]:


df.head()


# In[20]:


plt.figure(figsize=(10,6))
sns.boxplot(x='month', y='price', data=df)


# In[21]:


df.groupby('year').mean()['price'].plot()


# In[22]:


df = df.drop('date', axis=1)


# In[23]:


df.columns


# In[24]:


#df['zipcode'].value_counts()


# In[25]:


df = df.drop('zipcode',axis=1)


# In[26]:


X = df.drop('price',axis=1).values
y= df['price'].values


# In[27]:


X


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[30]:


from sklearn.preprocessing import MinMaxScaler


# In[31]:


scaler = MinMaxScaler()


# In[32]:


X_train = scaler.fit_transform(X_train)


# In[33]:


X_test = scaler.transform(X_test)


# In[34]:


import tensorflow as tf


# In[35]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation


# In[36]:


model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimzer='adam', loss='mse')


# In[37]:


model.fit(x=X_train, y=y_train, 
          validation_data=(X_test,y_test), 
         batch_size=128, epochs=400)


# In[38]:


losses = pd.DataFrame(model.history.history)


# In[39]:


losses.plot()


# In[40]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[41]:


predictions = model.predict(X_test)


# In[42]:


mean_squared_error(y_test, predictions)**0.5


# In[43]:


mean_absolute_error(y_test, predictions)


# In[45]:


df['price'].describe()


# In[46]:


explained_variance_score(y_test, predictions)


# In[47]:


plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
plt.plot(y_test,y_test,'r')


# Decent score for prediction considering its hitting the majority of houses

# Maybe worth, training on the bottom 99%

# In[48]:


single_house = df.drop('price', axis=1).iloc[0]


# In[49]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# Dropping first house in the DF to test model to see how well it would perform if house 1 was a new house

# In[50]:


model.predict(single_house)


# In[51]:


df.head()


# Over shooting

# Could retrain to drop top 1-2% of the most expensive houses, not too far off but could be retrained

# In[52]:


from tensorflow.keras.models import load_model


# In[53]:


model.save('housetraning_model.h5')


# In[ ]:




