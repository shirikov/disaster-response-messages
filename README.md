# Analysis of disaster response messages (WORK IN PROGRESS)

This project provides a classifier for messages in the wake of natural disasters. The model uses text features to predict message categories. The analysis is conducted as part of the Udacity Data Scientist Nanodegree.

## Required software

Python 3 (e.g., an Anaconda installation) with the following libraries installed: pandas, numpy, sklearn, ast, nltk, requests, bs4, matplotlib, datetime, sqlalchemy.

## Data description

The data set is constructed from two original data sets, where the first provides translations for a variety of social media messages posted during various disasters, and the second provides category labels for the messages (data compiled by Figure Eight). There are 36 message categories, such as "medical help," "search and rescue," etc. (the classifier predicts 35 categories, as there are no messages labeled as "child alone").

The data sets are combined and cleaned using the script **process_data.py**.

## Modeling

Several machine learning algorithms are applied to the resulting data set via the script **train_classifier.py**:

1. The messages are tokenized and transformed into a TF-IDF matrix.
2. A classifier is fitted to the matrix.
3. For some models, additional parameter tuning via grid search is done.
4. The best model is fitted with an additional feature, which is message sentiment (polarity scores).
5. The final model is exported as a pickle file for subsequent visualization.

Classifiers used in the analysis are: Naive Bayes; logistic regression; random forest. All performed rather similarly, with overall prediction accuracy above 93 percent, but the logit model turned out to be slightly better (almost 95 percent accuracy).

## Visualization (web app)

WORK IN PROGRESS

## Acknowledgments

The data used in the analysis are available from Udacity and Figure Eight. The code provided in this repo can be used and modified as needed.