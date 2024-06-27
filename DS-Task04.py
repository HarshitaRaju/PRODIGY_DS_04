#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


training_data_path = r'C:\Users\harsh\twitter_training.csv'
training_data = pd.read_csv(training_data_path)
training_data.head()


# In[3]:


training_data.columns = ['ID', 'Entity', 'Sentiment', 'Tweet']
training_data.head()


# In[4]:


missing_values = training_data.isnull().sum()
print(missing_values)


# In[5]:


def clean_tweet(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text

training_data['Cleaned_Tweet'] = training_data['Tweet'].apply(clean_tweet)
training_data.head()


# In[6]:


def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

training_data['Predicted_Sentiment'] = training_data['Cleaned_Tweet'].apply(get_sentiment)
training_data.head()


# In[12]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Predicted_Sentiment', hue='Predicted_Sentiment', data=training_data, palette=['#FF5733', '#33FF57', '#3357FF'], dodge=False, legend=False)
plt.title('Sentiment Distribution')
plt.show()


# In[10]:


processed_data_path =r'C:\Users\harsh\processed_twitter_training.csv'
training_data.to_csv(processed_data_path, index=False)


# In[ ]:




