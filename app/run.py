import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

app = Flask(__name__)

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

# Load the data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# Load the model
model = load('../models/classifier.pkl')

# Index webpage visualizes the data and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract the data needed for visuals
	# Count the number of messages in each 'genre'
    count_by_genre = df.groupby('genre').count()['message']
    genre_names = list(count_by_genre.index)
	
	# Count the number of messages for each label (category)
    count_by_category = df.drop(
        columns=['id', 'message', 
                 'original', 'genre']).sum().sort_values(ascending=False)
    category_names = list(count_by_category.index)

    # Create the visuals
    graphs = [
        {
            'data': [Bar(x=category_names, y=count_by_category)],
            'layout': {
                'template': 'plotly-dark',
                'title': 'Distribution of Messages by Category',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': '', 'tickangle': -45},
                'paper_bgcolor': 'LightSteelBlue'
                }
            },
        {
            'data': [Bar(x=genre_names, y=count_by_genre)],
            'layout': {
                'title': 'Distribution of Messages by Genre',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': ''},
                'paper_bgcolor': 'LightSteelBlue'
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use the model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html  
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()