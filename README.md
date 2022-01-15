# Analysis of disaster response messages (WORK IN PROGRESS)

This project provides a classifier for messages in the wake of natural disasters. The model uses text features to predict message categories. The analysis is conducted as part of the Udacity Data Scientist Nanodegree.

## Required software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, nltk, matplotlib, sqlalchemy, string, joblib, plotly, flask.

## Overview of the code

The initial data are cleaned up using the script **data/process_data.py**, then a classifier is trained and tested using **models/train_classifier.py**.

## Data description and cleanup

The data set is constructed from two original data sets, where **messages.csv** provides translations for a variety of messages posted during various disasters, and **categories.csv** provides category labels for the messages (data were compiled by Figure Eight). There are 36 message categories, such as "medical help," "search and rescue," etc.; the classifier is applied to 35 categories, as there are no messages labeled "child alone". In total, the cleaned-up data include 26,248 messages.

The script **data/process_data.py** cleans the data set and saves it in an SQL database. To run the script, execute the following:

*'python data/process_data.py data/messages.csv data/categories.csv data/disaster_messages.db'*

## Modeling

The script **models/train_classifier.py** then applies a text classifier to the cleaned data:

1. The messages are tokenized and transformed into a TF-IDF matrix.
2. A classifier is fitted to the matrix.
3. Additional parameter tuning via grid search is done.
4. The final model is exported as a pickle file for the web app.

To run the script, execute the following:

*'python models/train_classifier.py data/disaster_messages.db models/classifier.pkl'*

The main classifier used is a random forest model. An additional script **explore_classifiers.py** tries two other classifiers, Naive Bayes and SVM, as well as an additional feature (message sentiment). These alternatives do not improve predictions.

## Results

All classifiers performed on average rather similarly, with overall prediction accuracy above 93 percent, but the random forest model turned out to be slightly better (over 94 percent accuracy).

However, it is worth noting the imbalance in the data---there is a small number of messages in some of the categories. In such cases, negatives (**not falling** into a particular category) are predicted better than positives (falling into that category). For example, for the category "medical help" (only about 10 percent of all messages are labeled as "medical help") both precision and recall are close to 1 when predicting that a message does not fall under medical help, but both are much lower when predicting that it does. Predicting becomes even more difficult as the number of "successes" (positives) decreases. E.g., only a few dozen messages out of 26 thousand are labeled as "offer," and the classifier fails here: none of the messages are predicted to be in that category. Relatedly, we can see that when there are just a few positives, recall is much lower than precision. In other words, while the classifier rarely produces false positives, it tends to produce some false negatives.

One way to address this problem could be to fit different classifiers to different categories, tuning some of them to predict rare positives. However, given that in some categories, we have only a few messages, it would probably make more sense to merge these categories with some related ones (unless a particular category is really important to us, and it is crucial to try to predict such messages).

## Web app: classification and visualization

WORK IN PROGRESS

## Acknowledgments

The data used in the analysis are available from Udacity and Figure Eight. The code provided in this repo can be used and modified as needed.