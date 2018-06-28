#!/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer     # bag of words tokenizer
from sklearn.feature_extraction.text import TfidfTransformer    # occurances to frequencies -> term frequencies (occurance of word/total words)
from sklearn.naive_bayes import MultinomialNB                   # Naive Bayes classifier
from sklearn.linear_model import SGDClassifier                  # SVM classifier
from sklearn.pipeline import Pipeline                           # build a pipeline
import numpy as np                                              # eval performance
from sklearn import metrics                                     # detailed performance metrics
from sklearn.model_selection import GridSearchCV                # grid search for parameter tuning

def getTrainingDocs(file = "trainingdata.txt"):
    '''simple read file, split records and categories'''
    # data = np.genfromtxt(file, delimiter=',', dtype=str)  # alt file read with np
    with open(file, 'r') as infile: data=infile.read()
    training_records = data.strip().split('\n')     # training file record separator is \n, first remove any trailing new line '\n'
    num_docs = training_records.pop(0)              # first line is num/docs
    training_docs = {'target':[], 'data':[]}        # setup dictionary with documents and target
    for record in training_records:
        category, doc = record.split(' ', 1)
        training_docs['target'].append(category)
        training_docs['data'].append(doc)
    return(num_docs, training_docs)

if __name__ == '__main__':
    num_docs, training_docs = getTrainingDocs()

    # build a pipeline with Naive Bayes: vectorizer -> transformer -> classifier
    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

    # train with SVM classifier
    text_clf = Pipeline([
        ('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),])

    # train model using pipeline
    text_clf.fit(training_docs['data'], training_docs['target'])

    # new documents to predict
    t = int(input())
    test_docs = []
    for t_itr in range(t):
        test_docs.append(input().lower())
    predicted = text_clf.predict(test_docs)
    for doc, category in zip(test_docs, predicted):
        print('{1}'.format(doc, category))
        # print('{1} => {0}'.format(doc, category))
