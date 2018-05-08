import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

# TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    #raise NotImplementedError()
    w = w.reshape(13, 1)
    L = np.zeros_like(X)

    #w.reshape(13, 1)
    for i in range(len(X)):
        X0 = X[i].reshape(1,13)
        p = np.dot(X0, w) - y[i]
        L[i] = 2 * p * X0
    gradient = np.mean(L, axis=0)

    return gradient

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage

    true_grad = lin_reg_gradient(X, y, w)
    print('true_grad: ',true_grad)

    k = 500
    grad = np.zeros([k, 13])
    for i in range(k):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        grad[i] = batch_grad

    batch_grad_mean = np.mean(grad, axis=0)

    square_distance = np.mean((true_grad - batch_grad_mean) ** 2)
    cosine = cosine_similarity(true_grad, batch_grad_mean)
    print(square_distance, cosine)

    m = 400


    var_array = np.zeros([m, 13])

    for j in range(1, m):
        print(j)
        for i in range(k):
            X_b, y_b = batch_sampler.get_batch(m=j)
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            grad[i] = batch_grad

        batch_grad_var = np.var(grad, axis=0)
        var_array[j] = batch_grad_var

    var_log = np.log(var_array)
    M = np.arange(1, m+1)

    plt.scatter(np.log(M), var_log[:, 1])
    plt.xlabel("logm")
    plt.ylabel("log $sigma$")
    plt.title("log $sigma$ vs logm")

    plt.show()









if __name__ == '__main__':
    main()
