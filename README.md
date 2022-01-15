# Analysis of disaster response messages (WORK IN PROGRESS)

This project provides a classifier for messages in the wake of natural disasters. The model uses text features to predict message categories. The analysis is conducted as part of the Udacity Data Scientist Nanodegree.

## Required software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, nltk, matplotlib, sqlalchemy, string, joblib, plotly, flask.

## Overview of the code

The initial data are cleaned up using the script **data/process_data.py**, then a classifier is trained and tested using **models/train_classifier.py**.

## Data description and cleanup

The data set is constructed from two original data sets, where **messages.csv** provides translations for a variety of messages posted during various disasters, and **categories.csv** provides category labels for the messages (data were compiled by Figure Eight). There are 36 message categories, such as "medical help," "search and rescue," etc.; the classifier is applied to 35 categories, as there are no messages labeled "child alone". In total, the cleaned-up data include 26,248 messages.

The script **data/process_data.py** cleans the data set and saves it in an SQL database. To run the script, execute the following:

*'python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db'*

## Modeling

The script **models/train_classifier.py** then applies a text classifier to the cleaned data:

1. The messages are tokenized and transformed into a TF-IDF matrix.
2. A classifier is fitted to the matrix.
3. Additional parameter tuning via grid search is done.
4. The final model is exported as a pickle file for the web app.

To run the script, execute the following:

*'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'*

The main classifier used is a support vector machine classification. An additional script **explore_classifiers.py** tries two other classifiers, Naive Bayes and random forests, as well as an additional feature, message sentiment. (To look at these alternatives, execute *'python models/explore_classifiers.py data/DisasterResponse.db'*.)

## Results

Judging by the F1 score, SVM produced better predictions. Prediction quality, however, varies substantially between categories of messages. For example, weather-related messages are predicted fairly well, whereas requests for electricity or medical help are predicted very poorly. Partly this can be explained by an imbalance in the data---there are very few messages in some of these categories. As a result, in such cases, negatives (*not falling* into a particular category) are predicted much better than positives (falling into that category). E.g., only a few dozen messages out of 26 thousand are labeled as "offer," and the classifier completely fails here.

One way to address this problem could be to fit different classifiers to different categories, tuning some of them to predict rare positives. However, given that in some categories, we have only a few messages, any such improvements would be limited, and it would probably make more sense to merge these categories with some related ones (unless a particular category is really important to us, and it is crucial to try to predict such messages).

## Web app: classification and visualization

WORK IN PROGRESS

## Acknowledgments

The data used in the analysis are available from Udacity and Figure Eight. The code provided in this repo can be used and modified as needed.