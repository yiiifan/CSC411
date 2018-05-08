'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix as cm
from sklearn import metrics



def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('bow features')
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    shape = tf_idf_train.shape
    print('tf_idf feature')
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[tf_idf_train.sum(axis=0).argmax()]))

    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    scores = cross_val_score(model, bow_train, train_labels, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return model

def svm_clf(bow_train, train_labels, bow_test, test_labels):

    classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, n_iter=5, random_state=42)
    classifier.fit(bow_train, train_labels)

    train_pred = classifier.predict(bow_train)
    test_pred = classifier.predict(bow_test)

    print('SVM  train accuracy = {}'.format((train_pred == train_labels).mean()))
    print('SVM  test accuracy = {}'.format((test_pred == test_labels).mean()))

    confusion_matrix = np.zeros((20,20))
    for k in range(len(test_labels)):
        j = test_pred[k]
        i = test_labels[k]
        confusion_matrix[i][j] = confusion_matrix[i][j]+1
    print("Confusion Matrix : \n" ,confusion_matrix)


    print(metrics.classification_report(test_labels, test_pred))
    scores = cross_val_score(classifier, bow_train, train_labels, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



def NB_clf(bow_train, train_labels, bow_test, test_labels):

    classifier = MultinomialNB(alpha=0.2)
    classifier.fit(bow_train, train_labels)

    train_pred = classifier.predict(bow_train)
    print('Naive Bayes train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = classifier.predict(bow_test)
    print('Naive Bayes test accuracy = {}'.format((test_pred == test_labels).mean()))

    alpha= [0, 0.2, 0.4, 0.6, 0.8, 1]

    for i in range(len(alpha)):
        classifier = MultinomialNB(alpha=alpha[i])
        scores = cross_val_score(classifier, bow_train, train_labels, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def LogisticRegression_clf(bow_train, train_labels, bow_test, test_labels):

    classifier = LogisticRegression()
    classifier.fit(bow_train, train_labels)

    train_pred = classifier.predict(bow_train)
    print('LR train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = classifier.predict(bow_test)
    print('LR test accuracy = {}'.format((test_pred == test_labels).mean()))

    scores = cross_val_score(classifier, bow_train, train_labels, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    C = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
    for i in range(len(C)):
        classifier = LogisticRegression(C=C[i])
        scores = cross_val_score(classifier, bow_train, train_labels, cv=5)
        print(C[i])
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def cross_validation(train_data, train_labels, hyper):
    kf = sklearn.model_selection.KFold(n_splits=10)
    kf.get_n_splits(train_data)
    accur_train = np.zeros((len(hyper),))
    accur_test = np.zeros((len(hyper),))

    for k in range(len(hyper)):

        print('The hyperparameters is ', hyper[k])

        for train_index, test_index in kf.split(train_data):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=hyper[k], n_iter=5, random_state=42)
            classifier.fit(X_train, y_train)
            train_pred = classifier.predict(X_train)
            test_pred = classifier.predict(X_test)
            accur_train[k] = accur_train[k]+(train_pred == y_train).mean()
            accur_test[k] = accur_test[k]+ (test_pred == y_test).mean()

        accur_train[k] = accur_train[k]/10
        accur_test[k] = accur_test[k]/10

    print(accur_train,'and',accur_test)

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    train_tf_idf, test_tf_idf, feature_names_tf_idf = tf_idf_features(train_data, test_data)


    #bnb_model_bow = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    #bnb_model_tfidf = bnb_baseline(train_tf_idf, train_data.target, test_tf_idf, test_data.target)

    svm_model = svm_clf(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    #NB_model = NB_clf(train_tf_idf, train_data.target, test_tf_idf, test_data.target)
    #LR_model = LogisticRegression_clf(train_tf_idf, train_data.target, test_tf_idf, test_data.target)

    #tune hyperparameters alpha in svm
    #hyper = [1e-03, 1e-04, 1e-05,1e-06, 1e-07]
    #cross_validation(train_tf_idf, train_data.target, hyper)
