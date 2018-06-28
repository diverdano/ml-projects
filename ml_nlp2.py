#!/bin/python3

# from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups                 # data
from sklearn.feature_extraction.text import CountVectorizer     # bag of words tokenizer
from sklearn.feature_extraction.text import TfidfTransformer    # occurances to frequencies -> term frequencies (occurance of word/total words)
from sklearn.naive_bayes import MultinomialNB                   # Naive Bayes classifier
from sklearn.linear_model import SGDClassifier                  # SVM classifier
from sklearn.pipeline import Pipeline                           # build a pipeline
import numpy as np                                              # eval performance
from sklearn import metrics                                     # detailed performance metrics
from sklearn.model_selection import GridSearchCV                # grid search for parameter tuning

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# bag of words representation - tokenize
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

# dictionary of feature indices
count_vect.vocabulary_.get(u'algorithm')

# downscaling Term Frequency time Inverse Document Frequency
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)    # first fit
X_train_tf = tf_transformer.transform(X_train_counts)                   # then transform
X_train_tf.shape

# alternate approach - combining fit & transofrm
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)         # fit and transform
X_train_tfidf.shape

# training with Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# new documents to predict
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)                           # only transform, since model fit with training data
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('{0} => {1}'.format(doc, twenty_train.target_names[category]))

# build a pipeline: vectorizer -> transformer -> classifier
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])

# train model with a single command
text_clf.fit(twenty_train.data, twenty_train.target)

# evaluation of performance
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

# train with SVM classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)

# detailed performance
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
metrics.confusion_matrix(twenty_test.target, predicted)

# parameter tuning using grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

# perform on smaller set of data
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("{0}: {1}".format(param_name, gs_clf.best_params_[param_name]))

# more detailed results
gs_clf.cv_results_
