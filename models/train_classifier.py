import sys
import pandas as pd
from joblib import dump
from sqlalchemy import create_engine
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

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

def build_model():
    
    '''
    Construct the pipeline and set parameters for grid search.
    
    Returns: model pipeline
    '''
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(SVC()))
    ])

    parameters = {
            'tfidf__ngram_range': ((1, 1), (1, 2)),
            'clf__estimator__C': [0.01, 1, 100],
            'clf__estimator__kernel': ['poly', 'rbf']
        }
    pipeline_cv = GridSearchCV(pipeline, param_grid=parameters,
                               n_jobs=-1, verbose=3)
    
    return(pipeline_cv)

def evaluate_model(model, X_test, Y_test):
    
    '''
    Evaluate model fit on the test data.
    
    Args:
        model: model pipeline
        X_test: messages from test data
        Y_test: category labels from test data
    '''
    
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
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
        
def save_model(model, model_filepath):
    
    '''
    Save the model in a pickle file.
    
    Args:
        model: fitted model
        model_filepath: path to a file where a classifier is saved
    '''
    
    dump(model.best_estimator_, model_filepath)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size=0.3,
                                                            random_state=42)
        
        print('Building the model...')
        model = build_model()
        
        print('Training the model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating the model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving the model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
 