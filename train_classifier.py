import pandas as pd
import numpy as np
from joblib import dump
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Define a custom tokenizer
def tokenize(text):
    
    '''Process and tokenize messages.'''
    
    # Remove punctuation and transform to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# Load SQL database
engine = create_engine('sqlite:///disaster_messages.db')
df = pd.read_sql_table('disaster_messages', engine)

# Remove columns with no variation (nothing to predict)
df = df[df.columns[df.nunique() > 1]]

# Split into predictor and outcome variables
X = df.message
Y = df.drop(df.columns[[0, 1, 2, 3]], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, 
                                                    random_state=42)

# Pipeline with a Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)),
    ('clf', MultiOutputClassifier(BernoulliNB()))
])

# Fit on training data
pipeline.fit(X_train, Y_train)

# Predict on test data
Y_pred = pipeline.predict(X_test)
(Y_pred == Y_test).mean().mean()
(Y_pred == Y_test).mean()

# Improving model fit via grid search
parameters = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.5, 0.75, 1.0),
        'tfidf__max_features': (None, 5000, 10000),
        'clf__estimator__alpha': [0.25, 0.5, 0.75, 1]
    }

cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, Y_train)
cv.predict(X_test)
Y_pred = cv.predict(X_test)
(Y_pred == Y_test).mean().mean()

# Different models: logistic, random forests (without MultiOutput)
pipeline2 = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)),
    ('clf', MultiOutputClassifier(LogisticRegression()))
])

pipeline3 = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)),
    ('clf', RandomForestClassifier())
])

# Fit on training data
pipeline2.fit(X_train, Y_train)
pipeline3.fit(X_train, Y_train)

# Predict on test data
Y_pred = pipeline2.predict(X_test)
(Y_pred == Y_test).mean().mean()
(Y_pred == Y_test).mean()

Y_pred = pipeline3.predict(X_test)
(Y_pred == Y_test).mean().mean()
(Y_pred == Y_test).mean()

# Other features - sentiment
class SentimentExtractor(BaseEstimator, TransformerMixin):

    def get_polarity_scores(self, text):
        '''Returns polarity scores for words in Airbnb comments.'''
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
    
pipeline_upd = Pipeline([
     ('features', FeatureUnion([

        ('tfidf', TfidfVectorizer(tokenizer=tokenize, 
                                  stop_words=stop_words)),
        ('sentiment', SentimentExtractor())
        ])),

    ('clf', RandomForestClassifier())
    ])

# Export the model as a pickle file
dump(cv.best_estimator_, 'disaster_messages_model.pkl')