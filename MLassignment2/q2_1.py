'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
import sklearn.model_selection
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        digit = None
        dist = KNearestNeighbor.l2_distance(self, test_point)
        dist = dist.tolist()
        if k == 1:
            dist_min_position = dist.index(min(dist))
            digit = self.train_labels[dist_min_position]
        else:
            i = 0
            inf = 100000
            vote = np.zeros((10))
            knn_position = []
            while i < k:
                dist_min_position = dist.index(min(dist))
                dist[dist_min_position] = inf
                digit = self.train_labels[dist_min_position]
                i = i+1
                for j in range(10):
                    if digit == j:
                        vote[j] = vote[j]+1

            vote = vote.tolist()
            digit = np.argmax(vote)

        return digit

def cross_validation(train_data, train_labels,  k_range=np.arange(1,15)):
    kf = sklearn.model_selection.KFold(n_splits=10)
    kf.get_n_splits(train_data)
    accur_train = np.zeros((15,))
    accur_test = np.zeros((15,))

    for k in k_range:
        # for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        print('The number of neighbors is ',k)
        cnt=0
        # for k in k_range:
        for train_index, test_index in kf.split(train_data):
            X_train, X_test = train_data[train_index], train_data[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            knn = KNearestNeighbor(X_train, y_train)
            accur_train[k] = accur_train[k]+classification_accuracy(knn, k, X_train, y_train)
            accur_test[k] = accur_test[k]+classification_accuracy(knn, k, X_test, y_test)
            cnt=cnt+1

        accur_train[k] = accur_train[k]/cnt
        accur_test[k] = accur_test[k] /cnt
        print('aver accuracy on train set is %f, on test set is %f' % (accur_train[k], accur_test[k]))

        #for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    cnt = 0
    size = eval_data.shape[0]

    for i in range(size):
        predicted_label = knn.query_knn(eval_data[i], k)
        if predicted_label == eval_labels[i]:
            cnt = cnt + 1
    accuracy = cnt/size

    return accuracy



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #build a simple K nearest neighbor classifier
    knn = KNearestNeighbor(train_data, train_labels)
    k1 = 1
    k2 = 15

    accur1_test = classification_accuracy(knn, k1, test_data, test_labels)
    accur1_train = classification_accuracy(knn, k1, train_data, train_labels)
    accur2_test = classification_accuracy(knn, k2, test_data, test_labels)
    accur2_train = classification_accuracy(knn, k2, train_data, train_labels)
    print('When K = 1, the train accuracy is %f, the test accuracy is %f'%(accur1_train,accur1_test))
    print('When K = 15, the train accuracy is %f, the test accuracy is %f'%(accur2_train,accur2_test))

    #Use 10fold cross validation
    cross_validation(train_data, train_labels, k_range=np.arange(1, 15))

    # Example usage:
    #predicted_label = knn.query_knn(test_data[8], 15)
    #print(test_labels[8])
    #print(predicted_label)

if __name__ == '__main__':
    main()