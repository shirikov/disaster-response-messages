import sys
import pandas as pd
import numpy as np
from joblib import dump
from sqlalchemy import create_engine
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Load data
def load_data(database_filepath):
    
    '''Load the database from SQL.'''
    
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
    
    '''Process and tokenize messages.'''
    
    # Remove punctuation and transform to lower case
    text = text.translate(str.maketrans('', '', 
                                        string.punctuation)).lower().strip()
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if 
              word not in stop_words]

    return tokens

def build_model():
    
    '''Construct the pipeline and set parameters for grid search.'''
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', RandomForestClassifier())
    ])
    
    return(pipeline)

def evaluate_model(model, X_test, Y_test):
    
    '''Evaluate model fit on the test data.'''
    
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
    print('Overall model accuracy:', (Y_pred == Y_test).mean().mean(), '\n')
  
    print('Precision and recall for particular categories:\n')
    for i in range(0, Y_test.shape[1]):
        print('Category:', Y_test.columns[i], '\n', 
              classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        
def save_model(model, model_filepath):
    
    '''Save the model in a pickle file.'''
    
    dump(model, model_filepath)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size=0.3,
                                                            random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
 