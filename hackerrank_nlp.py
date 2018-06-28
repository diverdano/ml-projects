#!/bin/python3

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

    # X_train_counts = vectorize(training_docs['data'])
    count_vect = CountVectorizer()                                        # bag of words - tokenize
    X_train_counts = count_vect.fit_transform(training_docs['data'])
    print('X_train_counts.shape: {}'.format(X_train_counts.shape))
    # downscaling Term Frequency times Inverse Document Frequency - combining fit & transofrm
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)         # fit and transform
    print('X_train_tfidf.shape: {}'.format(X_train_tfidf.shape))
    # training with Naive Bayes classifier
    # clf = MultinomialNB().fit(X_train_tfidf, training_docs['target'])

    # train with SVM classifier
    text_clf = Pipeline([
        ('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),])
    text_clf.fit(training_docs['data'], training_docs['target'])

    # new documents to predict
    t = int(input())
    test_docs = []
    for t_itr in range(t):
        test_docs.append(input().lower())
        # test_docs.append([word_tokenize(item) for item in input().lower().split()])
    print('test docs: {}'.format(test_docs))
    X_new_counts = count_vect.transform(test_docs)                           # only transform, since model fit with training data
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    # predicted = clf.predict(X_new_tfidf)
    predicted = text_clf.predict(test_docs)
    for doc, category in zip(test_docs, predicted):
        print('{1} => {0}'.format(doc, category))
