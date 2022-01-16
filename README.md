# Analysis of disaster response messages

This project provides a classifier for messages in the wake of natural disasters. The model uses text features to predict message categories. Then, users can input custom messages into a web application and see how well the classifier predicts the categories of these messages. The analysis is conducted as part of the Udacity Data Scientist Nanodegree.

## Required software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, nltk, json, sqlalchemy, string, joblib, plotly, flask.

## Overview of the code

The initial data are cleaned up using the script **data/process_data.py**, then a classifier is trained and tested using **models/train_classifier.py**. Then, users can visualize the data and predict categories for a custom message via **app/run.py**. See below for details on how to run the code.

## Data description and cleanup

The data set is constructed from two original data sets, where **messages.csv** provides translations for a variety of messages posted during various disasters, and **categories.csv** provides category labels for the messages (data were compiled by Figure Eight). There are 36 message categories, such as "medical help," "search and rescue," etc.; the classifier is applied to 35 categories, as there are no messages labeled "child alone". In total, the cleaned-up data include 26,248 messages.

The script **data/process_data.py** merges the two initial data files, cleans the resulting data set, and saves it in an SQL database. To run the script, execute the following:

*python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db*,

where the last parameter is the name of the SQL database.

## Modeling

The script **models/train_classifier.py** then applies a text classifier to the cleaned data:

1. The messages are tokenized and transformed into a TF-IDF matrix.
2. A classifier is fitted to the matrix, with additional parameter tuning via grid search.
3. The final model is exported as a pickle file for the web app.

To run the script, execute the following:

*'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'* (it can take a while)

The main classifier used is a support vector machine classification. An additional script **explore_classifiers.py** tries two other classifiers, Naive Bayes and random forests, as well as an additional feature---message sentiment (to look at these alternatives, execute: *'python models/explore_classifiers.py data/DisasterResponse.db'*).

## Results

While the performance of different classifiers varied by message category, SVM has produced better predictions overall, based on F1 scores. Even with a better classifier, however, prediction quality varies quite a lot depending on the category. For example, weather-related messages are predicted fairly well, whereas requests for electricity or medical help are predicted poorly. Partly this can be explained by an imbalance in the data. There are very few messages in some of these categories, and in such cases, negatives (*not falling* into a particular category) are predicted much better than positives (falling into that category). E.g., only a few dozen messages out of 26 thousand are labeled as "offer," and the classifier completely fails here.

One way to address this problem could be to fit and tune different classifiers to different categories. However, given that in some categories, we have very few messages, the potential for improvements is most likely limited. It might make more sense to merge such categories with related ones (unless a particular category is really important to us, and it is crucial to try to predict such messages).

## Web app: classification and visualization

The web app, implemented in Flask, provides some descriptive visuals about the data set (the prevalence of message categories and "genres") and predicts categories of custom messages that a user can input. The app should be run as follows:

1. Change the directory to 'app' (e.g., run 'cd app' in the terminal).
2. Execute *'python run.py'*.
3. Go to *http://0.0.0.0:3001/*, where '0.0.0.0' is a placeholder; if the app is launched on a local machine, the address should be *http://localhost:3001/*.

The app assumes that the database is saved as DisasterResponse.db in the 'data' subfolder and that the model is saved as classifier.pkl in the 'models' subfolder. If the file names are different, edit the following lines in run.py:

*engine = create_engine('sqlite:///../data/DisasterResponse.db')*

*model = load('../models/classifier.pkl')*

## Acknowledgments

The data used in the analysis are available from Udacity and Figure Eight. The code provided in this repo can be used and modified as needed.