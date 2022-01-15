import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, precision_score, recall_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Load data
def load_data(database_filepath):
    
    '''
    Load the database from SQL.
    
    Args:
        database_filepath: path to an SQL database
        
    Returns: data frame with text features, data frame with category labels
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)

    # Remove columns with no variation (nothing to predict)
    df = df[df.columns[df.nunique() > 1]]
    
    # Split into predictor and outcome variables
    X = df['message']
    Y = df.drop(df.columns[[0, 1, 2, 3]], axis=1)
    
    return X, Y

# Define a custom tokenizer
def tokenize(text):
    
    '''Process and tokenize messages.
    
    Args:
        text: text message
        
    Returns: a list of cleaned tokens
    '''
    
    # Remove punctuation and transform to lower case
    text = text.translate(str.maketrans('', '', 
                                        string.punctuation)).lower().strip()
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if 
              word not in stop_words]

    return tokens

# Load and split data
# This takes database file name as a command line argument
X, Y = load_data(str(sys.argv[1]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, 
                                                    random_state=42)

# Function to fit and test the model
def fit_test(model_pipeline, model_name):
    
    '''
    Fit the model and print out model evaluations on test data.
    
    Args:
        model_pipeline: a pipeline/a grid search object
        model_name: model name to display
    '''
    
    # Fit the model
    print('Fitting the model: ' + model_name)
    model_pipeline.fit(X_train, Y_train)
    print('Model fitted.')
    
    # Test the model
    Y_pred = model_pipeline.predict(X_test)
    
    print('Overall model accuracy:', (Y_pred == Y_test).mean().mean(), '\n')
    
    precision_cat = []
    recall_cat = []
    f1_cat = []
    
    for i, category in enumerate(Y_test.columns):
        precision_cat.append(precision_score(Y_test[category], Y_pred[:, i]))
        recall_cat.append(recall_score(Y_test[category], Y_pred[:, i]))
        f1_cat.append(f1_score(Y_test[category], Y_pred[:, i]))
        
    prec_rec = pd.DataFrame(
        {
            "category": Y_test.columns,
            "Precision": precision_cat,
            "Recall": recall_cat,
            "F1": f1_cat,
        }
    )
    
    print('Precision and recall for particular categories:\n')
    print(prec_rec)

# Naive Bayes classifier
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(BernoulliNB()))
])

fit_test(pipeline_nb, 'Naive Bayes')    

# Trying other models
# Support vector classification - substantially better
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(SVC()))
])

fit_test(pipeline_svc, 'SVM') 

# Random forest does substantially worse
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', RandomForestClassifier())
])

fit_test(pipeline_rf, 'Random forest') 

# Additional feature: message sentiment
# Custom transformer to add to the pipeline
class SentimentExtractor(BaseEstimator, TransformerMixin):

    def get_polarity_scores(self, text):
        '''Returns polarity scores for words in messages.'''
        try:
            text_sent = sia.polarity_scores(text)
            return text_sent['compound']
        except:
            return np.nan
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_sentiment = pd.Series(X).apply(self.get_polarity_scores)
        return pd.DataFrame(X_sentiment)

# Updated pipeline with the sentiment feature
pipeline_svc_upd = Pipeline([
     ('features', FeatureUnion([

        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('sentiment', Pipeline([
            ('sent_extract', SentimentExtractor())
            ]))
        ])),
     
     ('clf', MultiOutputClassifier(SVC()))
    ])

# Almost no improvement in accuracy with sentiment added
fit_test(pipeline_svc_upd, 'SVM + sentiment score')

# Going back to the SVM model without sentiment
# Grid search for optimal parameters for the TD-IDF matrix and 
# for the classifier 
parameters = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'clf__C': [0.01, 1, 100],
        'clf__kernel': ['poly', 'rbf']
    }
svc_cv = GridSearchCV(pipeline_svc, param_grid=parameters)
fit_test(svc_cv, 'SVM with parameter tuning')

# Check the parameters of the best model
svc_cv.best_params_
svc_cv.best_estimator_
