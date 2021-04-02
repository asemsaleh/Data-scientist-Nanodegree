#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


# In[3]:


# load data from database
engine = create_engine('sqlite:///disaster_pipeline.db')
df = pd.read_sql('select * from disaster_data',engine)
X = df['message']
Y = df.iloc[:,4:]


# In[4]:


#Define a function for computing f1 score
'''The F1 score can be interpreted as a weighted average of the precision and recall, 
where an F1 score reaches its best value at 1 and worst score at 0. \
The relative contribution of precision and recall to the F1 score are equal.'''
def multioutput_fscore(y_true,y_pred,beta=1):
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score


# ### 2. Write a tokenization function to process your text data

# In[5]:



def tokenize(text):
    """
    A tokenization function that finds and cleab urls
    INPUT: text
    OUTPUT: list of cleaned urls
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline= Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


#Splitign the data for training and validation data 
X_train, X_test, y_train, y_test = train_test_split(X, Y)
#Fit the pipline model with it 
pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


y_pred = pipeline.predict(X_test)


# In[9]:


multi_f1 = multioutput_fscore(y_test,y_pred, beta = 1)
overall_accuracy = (y_pred == y_test).mean().mean()

print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))


# In[10]:


y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
for column in y_test.columns:
    print(classification_report(y_test[column],y_pred_pd[column]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[11]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# In[12]:


pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
parameters  = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.75, 1.0),
    'features__text_pipeline__vect__max_features': (None, 5000),
    'features__text_pipeline__tfidf__use_idf': (True, False),
}
scorer = make_scorer(multioutput_fscore,greater_is_better = True)

cv = GridSearchCV(pipeline, param_grid=parameters, scoring = scorer,verbose = 2, n_jobs = -1)
cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file

# In[14]:


# save the model to disk
filename = 'classifier.sav'
pickle.dump(pipeline, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


# In[15]:


filename = 'classifie.sav'
pickle.dump(pipeline, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




