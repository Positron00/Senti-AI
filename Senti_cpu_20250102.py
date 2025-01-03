#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('vader_lexicon')

plt.style.use('ggplot')


# In[ ]:


#from huggingface_hub import notebook_login
#notebook_login()


# In[2]:


from datasets import load_dataset_builder
ds_builder = load_dataset_builder("imdb")

# Inspect dataset features
ds_builder.info.features


# In[3]:


# Load the IMDb dataset from Hugging Face
from datasets import load_dataset
imdb = load_dataset("imdb")


# In[4]:


# check the dataset split names
from datasets import get_dataset_split_names
get_dataset_split_names("imdb")


# In[5]:


imdb


# In[6]:


ds_train = imdb['train']
ds_train


# In[7]:


# EDA
# check the distribution of the labels
labels_train = pd.Series(ds_train['label'])
ax = labels_train.value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Sentiment',
          figsize=(10, 5))
ax.set_xlabel('Sentiment')
plt.show()


# In[8]:


# basic NLTK preprocessing
example_text = ds_train[0]['text']
example_text
tokens = nltk.word_tokenize(example_text)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[9]:


# VADER (Valence Aware Dictionary and Sentiment Reasoner), ignoring contexts
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores(example_text)


# In[10]:


# Run the polarity score on the entire training dataset
results_train = {}
counter = 0
for row in tqdm(ds_train, total=len(ds_train)):
    text = row['text']
    results_train[counter] = sia.polarity_scores(text)
    counter += 1


# In[11]:


vaders = pd.DataFrame(results_train).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders


# In[12]:


df_train = pd.DataFrame(ds_train)
df_train = df_train.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df_train, how='left')
vaders


# In[13]:


# plot VADER scores by sentiment label
ax = sns.barplot(data=vaders, x='label', y='compound')
ax.set_title('Compound Score by Sentiment Label')
plt.show()


# In[14]:


# plot VADER sub-scores by sentiment label
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='label', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='label', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='label', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[15]:


# Use the Roberta pretrained model for sentiment analysis
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[16]:


# Get the model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[17]:


# Run the example text through the Roberta model and get the scores
encoded_text = tokenizer(example_text, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
scores_dict


# In[18]:


# Make a function to run the Roberta model on each entry
def polarity_scores_roberta(example):
    # Tokenize with truncation
    encoded_text = tokenizer(
        example, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


# In[19]:


# Test the function on the example text
polarity_scores_roberta(example_text)


# In[20]:


# Run the Roberta model on the training dataset
results_train_both = {}
counter = 0

for row in tqdm(ds_train, total=len(ds_train)):
    try:
        text = row['text']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        results_train_both[counter] = both
        counter += 1
    except RuntimeError:
        print(f'Broke for row {counter}')


# In[21]:


results_df = pd.DataFrame(results_train_both).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df_train, how='left')
results_df


# In[22]:


# compare the VADER and Roberta scores
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='label',
            palette='tab10')
plt.show()


# In[23]:


# sanity check the VADER and Roberta scores
results_df.query('label == 0') \
    .sort_values('roberta_pos', ascending=False)['text'].values[0]


# In[24]:


results_df.query('label == 0') \
    .sort_values('vader_pos', ascending=False)['text'].values[0]


# In[25]:


results_df.query('label == 1') \
    .sort_values('roberta_neg', ascending=False)['text'].values[0]


# In[26]:


results_df.query('label == 1') \
    .sort_values('vader_neg', ascending=False)['text'].values[0]


# In[27]:


# Use the Transformers Pipeline for sentiment analysis
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[28]:


sent_pipeline(example_text)


# In[29]:


sent_pipeline('I love sentiment analysis!')


# In[30]:


sent_pipeline('I hate having no gpus!')


# In[31]:


sent_pipeline('What a beautiful day! All my clothes got wet!')


# In[ ]:





# In[ ]:





# In[ ]:


# randomize the test set
ds_test = imdb['test']
ds_test


# In[ ]:


# split a validation set from the test set


# In[ ]:




