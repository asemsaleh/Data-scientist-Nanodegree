import json
import plotly
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
app = Flask(__name__)

def tokenize(text):
    '''
    Input: text
    Output: splited,cleaned with grouping inflected words-lemmarizer-
    This function read text, split it and then reverse each word to original form
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    ''' this class inhertes BaseEstimator,TransformerMixin it Custom Transformer that finds starting with verb or not

    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
def multioutput_fscore(y_true,y_pred,beta=1):
    #'''The F1 score can be interpreted as a weighted average of the precision and recall, 
    #where an F1 score reaches its best value at 1 and worst score at 0. 
    #The relative contribution of precision and recall to the F1 score are equal.
    #Returns a dataframe with columns: 
    #f1-score (average for all possible values of specific class)
    #precision (average for all possible values of specific class)
    #recall (average for all possible values of specific class)'''
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    aid_rel1 = df[df['aid_related']==1].groupby('genre').count()['message']
    aid_rel0 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(aid_rel1.index)

    # let's calculate distribution of classes with 1
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)

    #sorting values in ascending
    class_distr1 = class_distr1.sort_values(ascending = False)

    #series of values that have 0 in classes
    class_distr0 = (class_distr1 -1) * -1
    class_name = list(class_distr1.index)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name = 'Aid related'

                ),
                Bar(
                    x=genre_names,
                    y= aid_rel0,
                    name = 'Aid not related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=class_name,
                    y=class_distr1,
                    name = 'Class = 1'
                    #orientation = 'h'
                ),
                Bar(
                    x=class_name,
                    y=class_distr0,
                    name = 'Class = 0',
                    marker = dict(
                            color = 'rgb(212, 228, 247)'
                                )
                    #orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]

    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()