import sys
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
def multioutput_fscore(y_true,y_pred,beta=1):
    
   '''
    Input:y_true: real target datam y_pred:predicted data
    The F1 score can be interpreted as a weighted average of the precision and recall, 
    where an F1 score reaches its best value at 1 and worst score at 0. 
    The relative contribution of precision and recall to the F1 score are equal.
    Returns a dataframe with columns: 
    f1-score (average for all possible values of specific class)
    precision (average for all possible values of specific class)
    recall (average for all possible values of specific class)
    '''
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
#Custom Transformer that finds starting with verb or not
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
    #Return self nothing else to do here   
    def fit(self, X, y=None):
        return self
    #Method that describes what we need this transformer to do
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def load_data(database_filepath):
    """"
    Input: database-sqllite
    Reads data from database, create engine 
    Output:X:train data,Y:target data,category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from messages', engine)
    X = df['message']
    y = df.iloc[:, 4:39]
    y['related'].replace(2, 1, inplace=True)
    category_names = y.columns.tolist()
    return X, y, category_names


def tokenize(text):
    """
    A tokenization function that finds and cleab urls
    INPUT: text
    OUTPUT: cleaned urls, read text, split it and then reverse each word to original form
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

def build_model():
    '''
    Input: None
    Output: model
    Function for building pipeline and GridSearch
    '''
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
    #wraps scoring functions for use in GridSearchCV
    scorer = make_scorer(multioutput_fscore,greater_is_better = True)
    #Exhaustive search over specified parameter values for the estimator.
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring = scorer,verbose = 2, n_jobs = -1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model 
    Input: model, x_test: validation data, Y_test: validation target
            category_names: categories in dataframe
    Output: print the overall accuracy, and F1 score
    '''
    y_pred = model.predict(X_test)
    multi_f1 = multioutput_fscore(Y_test,y_pred, beta = 1)
    overall_accuracy = (y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    y_pred_pd = pd.DataFrame(y_pred, columns = Y_test.columns)
    #Loop over categories and build a text report showing the main classification metrics
    for column in Y_test.columns:
            print('------------------------------------------------------\n')
            print('FEATURE: {}\n'.format(column))
            print(classification_report(Y_test[column],y_pred_pd[column]))
def save_model(model, model_filepath):
    '''
    Input: trained model, path to be saved
    Output: save model into pickle file
    this function save the model into serilized file pickle
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        #database path
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()