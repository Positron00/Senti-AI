#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with Machine Learning

# ## Import the necessary libraries

# In[1]:


from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re
import string
import math
from tqdm.notebook import tqdm
from scipy.special import softmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset_builder, load_dataset, get_dataset_split_names, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('stopwords')
nltk.download('vader_lexicon')

plt.style.use('ggplot')


# ## Check the features of the IMDb dataset, then load the dataset

# In[2]:


ds_builder = load_dataset_builder("imdb")

# Inspect dataset features
ds_builder.info.features


# In[3]:


# Load the IMDb dataset from Hugging Face
imdb = load_dataset("imdb")


# In[4]:


# Check the dataset split names
get_dataset_split_names("imdb")


# In[5]:


# Check the dataset
imdb


# In[6]:


# Check the training dataset
ds_train = imdb['train']
ds_train


# In[7]:


# Check the test dataset
ds_test = imdb['test']
ds_test


# In[8]:


# Concatenate the training and test datasets so we can shuffle and split later
ds = concatenate_datasets([ds_train, ds_test])
ds


# ## Exploratory Data Analysis (EDA)

# In[9]:


# Use pandas to check for missing texts and labels
df = pd.DataFrame(ds) # convert the dataset to a pandas dataframe
num_missing_texts = df['text'].isna().sum()
print(f'Number of missing texts: {num_missing_texts}')

num_missing_labels = df['label'].isna().sum()
print(f'Number of missing labels: {num_missing_labels}')


# In[10]:


# Add a new column to the dataframe to store the word count of each text
df['length'] = df['text'].apply(lambda x: len(x.split()))
df['length'].describe()


# In[11]:


# Check the distribution of the review lengths (word count)
df['length'].plot(kind='hist', bins=100, figsize=(10, 5), title='Distribution of Review Lengths',\
                   xlabel='Word Count', ylabel='Frequency', color='blue')
plt.tight_layout()
plt.show()


# In[12]:


# Check the distribution of the sentiment labels
ax = df['label'].value_counts().sort_index() \
    .plot(kind='bar',color='blue',
          title='Count of Reviews by Sentiment',
          figsize=(10, 5))
ax.set_xlabel('Sentiment')
plt.show()


# ## Text Preprocessing

# In[13]:


# Check the first text in the dataset
example_text = df.iloc[0]['text']
example_text


# In[14]:


def preprocess_text(text: str) -> Tuple[List[str], str]:
    """
    Preprocess the text and remove unnecessary characters and stopwords.
    
    Input:
        text (str): The input text to preprocess
        
    Output:
        List[str]: A list of tokens after tokenizing, removing stopwords & punctuation, and stemming
        str: The processed text
    """
    # remove hyperlinks
    new_text = re.sub(r'https?://[^\s\n\r]+', '', text)
    # remove hashtags
    new_text = re.sub(r'#', '', new_text)
    # remove numbers
    new_text = re.sub(r'\d+', '', new_text)
    # tokenize the text
    tokens = nltk.word_tokenize(new_text.lower())
    # collect stopwords
    stop_words = set(stopwords.words('english'))
    # set up the stemmer
    stemmer = PorterStemmer()
    # remove stopwords & punctuation, then stem the words
    tokens = [stemmer.stem(word) for word in tokens \
              if word not in stop_words and word not in string.punctuation]
    processed_text = ' '.join(tokens)

    return tokens, processed_text


# In[15]:


# Test the preprocessing function
preprocess_text(example_text)


# In[16]:


# Apply the preprocessing function to the entire dataset
tqdm.pandas() # enable tqdm for pandas
df['tokens'], df['processed_text'] = zip(*df['text'].progress_apply(preprocess_text)) # apply the preprocessing function to the text column
df.head()


# In[17]:


# Add a new column to the dataframe to store the token counts of each text
df['token_count'] = df['tokens'].apply(len)
df['token_count'].describe()


# In[18]:


# Check the distribution of the token counts
df['token_count'].plot(kind='hist', bins=100, figsize=(10, 5), title='Distribution of Token Counts',\
                       xlabel='Token Count', ylabel='Frequency', color='blue')
plt.tight_layout()
plt.show()


# ### Build a dictionary of (token,label) frequencies

# In[19]:


# Build a dictionary of (token, label) frequencies
token_freqs = defaultdict(int)
for row in df.itertuples(index=True): # iterate over rows
    for token in row.tokens: # iterate over tokens per row
        token_freqs[(token, row.label)] += 1

token_freqs


# ### Feature Extraction

# In[20]:


# Define a function to extract token,label features from a list of tokens
def extract_features(tokens: List[str], token_freqs: Dict[Tuple[str, int], int]) -> np.ndarray:
    """
    Extract token,label features from a list of tokens.
    
    Input: 
        tokens: a list containing tokens from a text
        token_freqs: a dictionary corresponding to the frequencies of each tuple (token, label)
    Output: 
        x: a feature vector of dimension (1,2)
    """
    # Initialize the feature vector
    x = np.zeros((1, 2))

    # Iterate over the tokens and update the feature vector
    for token in tokens:
        # increment the token count for the positive label 1
        x[0,0] += token_freqs.get((token,1), 0)
        # increment the token count for the negative label 0
        x[0,1] += token_freqs.get((token,0), 0)

    return x


# In[21]:


# Apply the feature extraction function to the entire dataset
tl_features = np.concatenate([extract_features(tokens, token_freqs) for tokens in df['tokens']])
df[['tlc_pos','tlc_neg']] = tl_features
df.head()


# ## Model Building

# ### Shuffle and split the dataset

# In[22]:


# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=12).reset_index(drop=True)
shuffled_df.head()


# In[23]:


# Split the dataset into training and test sets
df_train = shuffled_df.iloc[:40000]
df_test = shuffled_df.iloc[40000:50000]

#df_train.head()
#df_test.head()


# ### Train a Logistic Regression model

# In[24]:


# Extract the features and labels
X_train = df_train[['tlc_pos','tlc_neg']].values
y_train = df_train['label'].values

X_test = df_test[['tlc_pos','tlc_neg']].values
y_test = df_test['label'].values


# In[25]:


""" Without hyperparameter tuning

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = classifier.predict(X_val)

# Get the confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_val, y_val_pred)
print(cm)
accuracy_score(y_val, y_val_pred)
"""


# In[26]:


# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with StandardScaler and LogisticRegression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Define the parameter grid
param_grid = {
    'logisticregression__C': np.logspace(-4, 4, 10),  # Regularization strength
    'logisticregression__penalty': ['l1', 'l2'],      # Regularization type
    'logisticregression__solver': ['liblinear', 'saga']  # Compatible solvers
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # use all processors
)

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)


# In[27]:


# Make predictions on the test set with the best model
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)


# In[28]:


# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

models = []
accuracy_scores = {}
f1_scores = {}

def evaluate_model(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    print('Confusion Matrix:')
    print(cm)
    print(f'Accuracy: {accuracy_score(y_test, y_test_pred)}')
    print('Classification Report:')
    print(classification_report(y_test, y_test_pred))
    print("Precision = %0.4f" % precision_score(y_test, y_test_pred))
    print("Recall = %0.4f" % recall_score(y_test, y_test_pred))
    print("F1 Score = %0.4f" % f1_score(y_test, y_test_pred))
    return accuracy_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)


# In[29]:


model = 'Logistic Regression'
models.append(model)
accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[30]:


# Error analysis
print('Truth Predicted Review')
for x, y_pred, y in zip(df_test['text'], y_test_pred, y_test):
    if y != y_pred:
        print('%d\t%0.2f\t%s' % (y, y_pred, ' '.join(
            x).encode('ascii', 'ignore')))


# ### Naive Bayes Model

# In[31]:


# Remake the token frequency dictionary from the training data
token_freqs = defaultdict(int)
for row in df_train.itertuples(index=True):
    for token in row.tokens:
        token_freqs[(token, row.label)] += 1

token_freqs


# In[32]:


def train_naive_bayes(token_freqs, y_train):
    '''
    Train a Naive Bayes model using the (token, label) frequencies and the training data.
    
    Input:
        token_freqs: dictionary from (token, label) to how often the token appears
        y_train: a list of labels corresponding to the tokens (0,1)
    Output:
        logprior: the log prior
        loglikelihood: the log likelihood of the Naive Bayes equation
    '''
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique tokens in the vocabulary
    vocab = [k[0] for k in token_freqs]
    V = len(set(vocab))    

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in token_freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            # Increment the number of positive tokens by the count for this (token, label) pair
            N_pos += token_freqs.get(pair, 0)

        # else, the label is negative
        else:
            # increment the number of negative tokens by the count for this (token,label) pair
            N_neg += token_freqs.get(pair, 0)
    
    # Calculate D, the number of documents
    D = len(y_train)

    # Calculate D_pos, the number of positive documents
    D_pos = np.sum(y_train==1)

    # Calculate D_neg, the number of negative documents
    D_neg = np.sum(y_train==0)

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)
    
    # For each token in the vocabulary...
    for token in vocab:
        # get the positive and negative frequency of the token
        freq_pos = token_freqs.get((token,1), 0)
        freq_neg = token_freqs.get((token,0), 0)

        # calculate the probability that each token is positive, and negative
        p_t_pos = (freq_pos+1)/(N_pos+V)
        p_t_neg = (freq_neg+1)/(N_neg+V)

        # calculate the log likelihood of the token
        loglikelihood[token] = np.log(p_t_pos) - np.log(p_t_neg)

    return logprior, loglikelihood

# Train the Naive Bayes model
logprior, loglikelihood = train_naive_bayes(token_freqs, y_train)
print(logprior)
print(len(loglikelihood))


# In[33]:


def naive_bayes_predict(tokens, logprior, loglikelihood):
    '''
    Input:
        tokens: a list of tokens
        logprior: a number for the log prior
        loglikelihood: a dictionary of tokens mapping to their log likelihood
    Output:
        logprob: the sum of all the logliklihoods of each token in the list (if found in the dictionary) + logprior (a number)
    '''

    # initialize probability to zero
    logprob = 0

    # add the logprior
    logprob += logprior  

    for token in tokens:
        # add the log likelihood of each token, if found, to the probability
        logprob += loglikelihood.get(token, 0)

    return logprob


# In[34]:


def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of list of tokens (reviews)
        test_y: the corresponding labels
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for the tokens
    Output:
        accuracy: (# of tokens classified correctly)/(total # of tokens)
    """
    accuracy = 0  # initialize accuracy

    y_pred = []
    for tokens in test_x:
        y_pred_i = 0 # default y_pred is 0

        # if the prediction is > 0, make the class 1
        if naive_bayes_predict(tokens, logprior, loglikelihood) > 0:
            y_pred_i = 1
            
        # append the predicted class to the list y_hats
        y_pred.append(y_pred_i)

    # accuracy is the percentage of correct predictions
    accuracy = np.sum(y_pred==test_y)/len(test_y)

    return accuracy, y_pred


# In[35]:


accuracy, y_test_pred = test_naive_bayes(df_test['tokens'], y_test, logprior, loglikelihood)

print("Naive Bayes accuracy = %0.4f" % accuracy)


# In[36]:


# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

model = 'Naive Bayes'
models.append(model)
accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[37]:


# Error analysis
print('Truth Predicted Review')
for x, y in zip(df_test['text'], y_test):
    logprob = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(logprob) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(logprob) > 0, ' '.join(
            x).encode('ascii', 'ignore')))


# In[38]:


# Multinomial Naive Bayes using Scikit-Learn
model = 'Multinomial Naive Bayes'
models.append(model)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

X_train = df_train['processed_text']
X_test = df_test['processed_text']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features

# Fit and transform training data; transform testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize Multinomial Naive Bayes model
mnb = MultinomialNB()

# Train the model
mnb.fit(X_train_tfidf, y_train)

# Predict on test data
y_test_pred = mnb.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[39]:


# Hyperparameter tuning of Multinomial Naive Bayes using GridSearchCV
model = 'Multinomial Naive Bayes GridSearchCV'
models.append(model)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define the parameter grid for alpha (smoothing)
param_grid = {'multinomialnb__alpha': [0.1, 0.5, 1.0]}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # use all processors
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters and accuracy score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Make predictions on the test set with the best model
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[40]:


# Hyperparameter tuning of Multinomial Naive Bayes using Bayesian Optimization
model = 'Multinomial Naive Bayes Bayesian Optimization'
models.append(model)

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from skopt.space import Real

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define the search space for MultinomialNB's alpha parameter (smoothing)
search_space = {
    'multinomialnb__alpha': Real(0.01, 1.0, prior='log-uniform')  # Search in log scale for better granularity
}

# Define Bayesian optimization with BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space,
    n_iter=30,  # number of iterations to sample from the search space
    cv=5,       # 5-fold cross-validation
    scoring='accuracy',  # metric to optimize (e.g., accuracy)
    n_jobs=-1,  # use all available processors
    random_state=42  # for reproducibility
)

# Fit the model
bayes_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", bayes_search.best_params_)
print("Best Cross-Validation Accuracy:", bayes_search.best_score_)

# Predict on test data using the best estimator from BayesSearchCV
best_model = bayes_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# ### Support Vector Machine (SVM) Model

# In[ ]:


# Hyperparameter tuning of SVM using RandomizedSearchCV, took 253 minutes to run!!!
model = 'SVM RandomizedSearchCV'
models.append(model)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from scipy.stats import uniform

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define hyperparameter distributions for RandomizedSearchCV
param_dist = {
    'C': uniform(0.1, 100),                # Uniform distribution for C between 0.1 and 100
    'gamma': uniform(0.001, 1),            # Uniform distribution for gamma between 0.001 and 1
    'kernel': ['linear', 'rbf']            # Kernel type
}

# Initialize RandomizedSearchCV with SVM model
random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist,
                                   n_iter=20, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the model on training data
random_search.fit(X_train_tfidf, y_train)

# Print best parameters and best score from RandomizedSearchCV
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)

# Get the best estimator from RandomizedSearchCV
best_model = random_search.best_estimator_

# Predict on test data
y_test_pred = best_model.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[41]:


# save the accuracy and f1 scores for SVM without running the hyperparameter tuning again
model = 'SVM RandomizedSearchCV'
models.append(model)
accuracy_scores[model] = 0.8881
f1_scores[model] = 0.8905


# In[ ]:


# Hyperparameter tuning of SVM using Ray Tune, not working yet
model = 'SVM Ray Tune'
models.append(model)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import ray
from ray import tune

# Initialize Ray locally
ray.init()

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


def train_svm(config):
    # Initialize the SVM model with hyperparameters from Ray Tune
    model = SVC(C=config["C"], gamma=config["gamma"], kernel=config["kernel"])

    # Implement k-fold cross-validation
    k_folds = 5  # Number of folds
    scores = []
    for fold in range(k_folds):
        # Split data into train and validation sets
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = \
            train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=fold, stratify=y_train)

        # Train the model on the training set
        model.fit(X_train_fold, y_train_fold)

        # Evaluate on the validation set
        predictions = model.predict(X_val_fold)
        validation_score = accuracy_score(y_val_fold, predictions)
        scores.append(validation_score)
    
    # Report average score across folds to Ray Tune
    tune.report(mean_accuracy=np.mean(scores))


search_space = {
    "C": tune.loguniform(1e-3, 1e3),  # regularization parameter
    "gamma": tune.loguniform(1e-4, 1e-1),  # kernel coefficient for RBF kernel
    "kernel": tune.choice(["linear", "rbf"])  # kernel type
}

# Run hyperparameter tuning with Ray Tune
analysis = tune.run(
    train_svm,
    config=search_space,
    num_samples=10,  # number of trials/samples from search space
    resources_per_trial={"cpu": 2},  # allocate CPU resources per trial; Scikit-learn runs on CPUs.
)

# Print best configuration and results
print("Best Config:", analysis.best_config)
print("Best Accuracy:", analysis.best_result["accuracy"])

# Retrieve best configuration from Ray Tune results
best_config = analysis.best_config

# Train final model with best parameters
best_model = SVC(C=best_config["C"], gamma=best_config["gamma"], kernel=best_config["kernel"])
best_model.fit(X_train_tfidf, y_train)

# Predict on test data
y_test_pred = best_model.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# ### Random Forest Model

# In[42]:


# Hyperparameter tuning of Random Forest using RandomizedSearchCV, takes 7 minutes to run
model = 'Random Forest RandomizedSearchCV'
models.append(model)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define hyperparameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(), 
    param_distributions=param_dist,
    n_iter=50, 
    cv=5, 
    scoring='accuracy', 
    verbose=2, 
    n_jobs=-1
)

# Fit the model on training data
random_search.fit(X_train_tfidf, y_train)

# Print best parameters and best score from RandomizedSearchCV
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)

# Get the best estimator from RandomizedSearchCV
best_model = random_search.best_estimator_

# Predict on test data
y_test_pred = best_model.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[ ]:


model = 'Random Forest RandomizedSearchCV'
models.append(model)
accuracy_scores[model] = 0.8567
f1_scores[model] = 0.8609


# In[ ]:


# Hyperparameter tuning of Random Forest using Ray Tune, not working yet
model = 'Random Forest Ray Tune'
models.append(model)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ray
from ray import tune

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

def train_random_forest(config):
    try:
        print(f"Starting training with config: {config}")

        # Initialize the Random Forest model with hyperparameters from Ray Tune
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            max_features=config["max_features"],
            random_state=42,
            n_jobs=-1  # use all available CPU cores
        )

        # Implement k-fold cross-validation
        k_folds = 5  # number of folds
        scores = []
        for fold in range(k_folds):
            # Split data into train and validation sets
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = \
                train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=fold, stratify=y_train)

            # Train the model on the training set
            model.fit(X_train_fold, y_train_fold)

            # Evaluate on the validation set
            predictions = model.predict(X_val_fold)
            validation_score = accuracy_score(y_val_fold, predictions)
            scores.append(validation_score)
        
        # Report average score across folds to Ray Tune
        tune.report(mean_accuracy=np.mean(scores))

    except Exception as e:
        print(f"Error in trial with config {config}: {e}")
        raise e  # mark the trial as failed

search_space = {
    "n_estimators": tune.randint(50, 300),  # number of trees in the forest
    "max_depth": tune.randint(5, 30),       # maximum depth of each tree
    "min_samples_split": tune.randint(2, 10),   # minimum fraction of samples required to split an internal node
    "max_features": tune.choice(["sqrt", "log2"])  # number of features considered for splitting
}

# Run hyperparameter tuning with Ray Tune
analysis = tune.run(
    train_random_forest,
    config=search_space,
    num_samples=5,  # number of trials/samples from search space
    resources_per_trial={"cpu": 1},  # allocate CPU resources per trial (Scikit-learn runs on CPUs)
    verbose=2,
)

# Print best configuration and results
print("Best Config:", analysis.best_config)
print("Best Accuracy:", analysis.best_result["accuracy"])

# Retrieve best configuration from Ray Tune results
best_config = analysis.best_config

# Train final model with best parameters
best_model = RandomForestClassifier(
    n_estimators=best_config["n_estimators"],
    max_depth=best_config["max_depth"],
    min_samples_split=best_config["min_samples_split"],
    max_features=best_config["max_features"],
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)
best_model.fit(X_train_tfidf, y_train)

# Predict on test data
y_test_pred = best_model.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# ### XGBoost Model

# In[ ]:


# Hyperparameter tuning of XGBoost using RandomizedSearchCV, takes 13 minutes to run
model = 'XGBoost RandomizedSearchCV'
models.append(model)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

# Convert text to numerical features using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define the XGBoost model and hyperparameter search space
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

param_distributions = {
    'n_estimators': randint(100, 500),        # number of boosting rounds
    'max_depth': randint(3, 10),              # maximum depth of a tree
    'learning_rate': uniform(0.01, 0.29),     # step size
    'subsample': uniform(0.5, 0.5),           # fraction of samples used for training each tree
    'colsample_bytree': uniform(0.5, 0.5),    # fraction of features used for each tree
    'min_child_weight': randint(1, 10),       # minimum sum of instance weights in a child node
    'gamma': uniform(0, 5)                    # minimum loss reduction required for a split
}

# Set up RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,                                # number of random combinations to try
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1,                                # parallelize across all available CPUs
    verbose=2,
    error_score='raise'
)

# Fit the model on training data
random_search.fit(X_train_tfidf, y_train)

# Print best parameters and best score from RandomizedSearchCV
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Accuracy:", random_search.best_score_)

# Get the best estimator from RandomizedSearchCV
best_model = random_search.best_estimator_

# Predict on test data
y_test_pred = best_model.predict(X_test_tfidf)

# Find precision, recall, and F1 score
# Get the confusion matrix, accuracy score, and classification report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[43]:


model = 'XGBoost RandomizedSearchCV'
models.append(model)
accuracy_scores[model] = 0.8741
f1_scores[model] = 0.8771


# ## Pretrained Models

# ### VADER (Valence Aware Dictionary and Sentiment Reasoner)

# In[44]:


# VADER (Valence Aware Dictionary and Sentiment Reasoner), ignoring contexts
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Run the polarity score on the test set, and unpack the scores into separate columns
tqdm.pandas()
vaders = df_test.progress_apply(lambda row: pd.Series(sia.polarity_scores(row['text'])), axis=1)
vaders = pd.DataFrame(vaders)
vaders.head()


# In[45]:


df_test.head()


# In[46]:


y_test_pred = vaders['compound'] > 0
model = 'VADER'
models.append(model)
accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# In[ ]:


# merge the VADER scores with the test set
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
df_test = df_test.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df_test, how='right')
vaders


# In[ ]:


# plot VADER scores by sentiment label
ax = sns.barplot(data=vaders, x='label', y='compound')
ax.set_title('Compound Score by Sentiment Label')
plt.show()


# In[ ]:


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


# ### Use the Roberta pretrained model for sentiment analysis

# In[47]:


# get the Roberta model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# try to run the model on Mac GPUs
# check if MPS (Metal Performance Shaders) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
model = model.to(device)

# Run the Roberta model with batch input using GPUs
def polarity_scores_roberta_gpu_batch(texts: List[str], batch_size: int = 256) -> Dict[int, Dict[str, float]]:
    """
    Analyze sentiment for a batch of texts using the Roberta model.
        
    Args:
        texts (List[str]): List of input texts
        batch_size (int): Size of the batch to process
            
    Returns:
        Dict[int, Dict[str, float]]: Dictionary of dictionaries containing sentiment probabilities
    """

    scores_dict = {}
    num_batches = math.ceil(len(texts) / batch_size)
         
    for i in tqdm(range(num_batches)):
        # Get the batch of texts
        batch_texts = texts[i*batch_size : (i+1)*batch_size]
        
        # Tokenize with truncation
        encoded_text = tokenizer(
                 batch_texts, 
                 return_tensors='pt', 
                 truncation=True, 
                 padding=True, 
                 max_length=512
             ).to(device)
             
        # Run inference
        with torch.no_grad():
            outputs = model(**encoded_text)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
             
        # Get predictions
        for j, probs in enumerate(probabilities.cpu().numpy()):
            idx = i * batch_size + j
            scores_dict[idx] = {
                     "roberta_neg": float(probs[0]),
                     "roberta_neu": float(probs[1]),
                     "roberta_pos": float(probs[2]),
                     "roberta_compound": np.log(float(probs[2])) - np.log(float(probs[0]))
                 }
         
    return scores_dict

roberta_batch_result = polarity_scores_roberta_gpu_batch(list(df_test['text']))
roberta_batch_result


# In[50]:


roberta_results = pd.DataFrame(roberta_batch_result).T
roberta_results


# In[51]:


y_test_pred = roberta_results['roberta_compound'] > 0
model = 'Roberta'
models.append(model)
accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# ### Use the Hugging Face pipeline for sentiment analysis

# In[55]:


# Use the Transformers Pipeline for sentiment analysis with truncation
from transformers import pipeline

# Initialize the sentiment analysis pipeline with truncation
sent_pipeline = pipeline(
    "sentiment-analysis",
    truncation=True,     # Enable truncation
    max_length=512       # Set maximum length to match model's requirement
)


# In[56]:


sent_pipeline(example_text)


# In[58]:


# Run the sentiment pipeline on the test set
tqdm.pandas()
hfpipe_results = df_test.progress_apply(lambda row: pd.Series(sent_pipeline(row['text'])[0]), axis=1)
hfpipe_results = pd.DataFrame(hfpipe_results)
hfpipe_results.head()


# In[59]:


y_test_pred = hfpipe_results['label'] == 'POSITIVE'
model = 'Hugging Face Pipeline'
models.append(model)
accuracy_scores[model], f1_scores[model] = evaluate_model(y_test, y_test_pred)


# # Compare the evaluation metrics of all the models

# In[62]:


# Convert score dictionaries to lists based on the order of 'models'
accuracy_list = [accuracy_scores[model] for model in models]
f1_list = [f1_scores[model] for model in models]

# Create the DataFrame
all_results = pd.DataFrame({
    'model': models,
    'accuracy': accuracy_list,
    'f1_score': f1_list
})

all_results.sort_values('f1_score', axis=0, ascending=True, inplace=True)
all_results


# In[75]:


# Plot the results as a grouped bar chart using seaborn
# Melt the DataFrame
df_melted = all_results.melt(id_vars="model", 
                              value_vars=["accuracy", "f1_score"], 
                              var_name="Metric", 
                              value_name="Score")


plt.figure(figsize=(12, 8)) # Initialize the matplotlib figure
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.color': 'gray'}) # Customize gridlines to gray
bar_plot =sns.barplot(x="model", y="Score", hue="Metric", data=df_melted, palette="Set2") # Create a barplot
plt.ylim(0.6, 1)  # Sets y-axis from 0.6 to 1

# Add value annotations on bars
for p in bar_plot.patches:
    height = p.get_height()
    if height > 0.5:
        bar_plot.annotate(f'{height:.2f}', 
                          (p.get_x() + p.get_width() / 2., height), 
                          ha='center', va='bottom', 
                          fontsize=12, color='black', 
                          xytext=(0, 5), 
                          textcoords='offset points')

# Add titles and labels
plt.title("Comparison of Accuracy and F1 Score Across Models", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("Scores", fontsize=14)

# Rotate x-axis tick labels by 45 degrees
plt.xticks(rotation=45, ha='right')  # ha='right' aligns labels to the right for better readability

# Customize legend
plt.legend(title="Metric", fontsize=12, title_fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()


# # Why is the SVM model so good?
# 
# ## **Key advantages:**
# 
# ### 1. **High Accuracy in Text Classification**
# SVMs consistently achieve high accuracy in text classification tasks, including sentiment analysis. This is attributed to their ability to handle complex decision boundaries and high-dimensional feature spaces.
# 
# ### 2. **Effective Handling of High-Dimensional Data**
# Text data, such as movie reviews, is inherently high-dimensional due to the large vocabulary size. SVMs are particularly well-suited for such scenarios because they can generalize well in high-dimensional spaces without requiring explicit feature selection.
# 
# ### 3. **Robustness to Overfitting**
# SVMs are designed to maximize the margin between classes, which helps them avoid overfitting, even when working with smaller datasets or sparse data (a common characteristic of text data).
# 
# ### 4. **Ability to Handle Sparse Data**
# Movie reviews often result in sparse feature vectors after preprocessing (e.g., using TF-IDF). SVMs excel in such cases by effectively finding hyperplanes that separate classes in sparse feature spaces.
# 
# ### 5. **Flexibility with Kernels**
# SVMs allow the use of different kernel functions (e.g., linear, polynomial, radial basis function), enabling them to model complex relationships in the data.
# 
# ### 6. **Integration with Preprocessing Techniques**
# SVMs work well with advanced preprocessing techniques like:
#    - **TF-IDF weighting**: Converts text into numerical features by emphasizing important words while reducing noise from frequently occurring terms.
#    - **SMOTE (Synthetic Minority Oversampling Technique)**: Balances datasets by addressing class imbalance issues.
# 
# 
# 
# ## **Limitations：**
# 
# ### 1. **Computationally Expensive**: 
# Training SVMs is computationally intensive, with complexity ranging from $$O(n^2)$$ to $$O(n^3)$$, making them impractical for very large datasets.
# ### 2. **Memory Intensive**: 
# SVMs require storing the entire dataset in memory during training, which can lead to significant overhead for large-scale datasets.
# ### 3. **Sensitive to Noise**: 
# Noisy data, such as slang, typos, or sarcasm in movie reviews, can mislead SVMs and reduce their accuracy.
# ### 4. **Limited Scalability**: 
# Due to their high computational and memory demands, SVMs are not well-suited for "big data" applications or datasets with millions of samples.
# 

# ## Retrieval Augmented Generation (RAG)

# In[ ]:





# In[ ]:




