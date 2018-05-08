import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

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

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr=1.0, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        v = params[0]
        w = params[1]
        v = self.beta * v - self.lr * grad
        w = w + v
        params = [v,w]
        return params

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)

        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss

        loss = 1-y*np.dot(X,self.w.reshape((-1,1)))
        loss  = 0.5 * ((self.w[1:] ** 2).sum()) + self.c * np.where(loss > 0, loss, 0).mean()
        return loss


    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        grad = np.zeros((X.shape[1],))

        for i in range(X.shape[0]):
            if 1-(y[i]*(np.dot(X[i],self.w.reshape(-1,1)))) >=0:
                grad = grad - y[i]*X[i]
        grad = self.w + (self.c * grad/X.shape[0])

        return grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        y_pred = np.zeros((len(X),))
        for i in range(len(X)):
            y_pred[i] = np.dot(X[i], self.w)
            if y_pred[i]>=0:
                y_pred[i] = 1
            else:
                y_pred[i]=-1

        return y_pred

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function( w_init=10.0, steps=200, lr=1.0, beta = 0.0):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    params = [0, w_init]
    w_history = np.zeros(steps)
    opt = GDOptimizer(lr, beta)

    for i in range(steps):
        # Optimize and update the history
        grad = func_grad(params[1])
        params = GDOptimizer.update_params(opt, params=params, grad=grad)
        w_history[i] = params[1]

    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    feature = train_data.shape[1]
    batch  = BatchSampler(train_data, train_targets, batchsize)
    svm = SVM(c=penalty, feature_count=feature )
    params = [0, svm.w]

    for j in range(iters):
        #calculate batch_grad
        X_batch, y_batch = batch.get_batch()
        batch_grad = svm.grad(X_batch, y_batch)
        params = optimizer.update_params(params=params, grad=batch_grad)
        svm.w = params[1]

    return svm

if __name__ == '__main__':

    #SGD with Momentum for 200 time-steps
    w1 = optimize_test_function(w_init=10.0, steps=200, beta=0.0)
    x = np.arange(0,200)
    plt.plot(x, w1)
    w2 = optimize_test_function(w_init=10.0, steps=200, beta=0.9)
    plt.plot(x, w2,'r--')
    plt.title("SGD with Momentum for 200 time-steps")
    plt.show()  

    penalty = 1.0
    batchsize = 100
    iters = 500

    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.concatenate((np.ones((train_data.shape[0],1)),train_data), axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0],1)), test_data), axis=1)
    feature = train_data.shape[1]

    #optimizer = GDOptimizer(lr=0.05, beta=0.0)
    optimizer = GDOptimizer(lr=0.05, beta=0.1)
    svm = optimize_svm(train_data, train_targets.reshape((-1,1)), penalty, optimizer, batchsize, iters)

    hinge_loss_train = svm.hinge_loss(train_data, train_targets.reshape((-1,1)))
    print('training loss  = ',hinge_loss_train)

    hinge_loss_test = svm.hinge_loss(test_data, test_targets.reshape((-1,1)))
    print('test loss  = ',hinge_loss_test)

    train_pred = svm.classify(train_data)
    print('train accuracy = {}'.format((train_pred == train_targets).mean()))

    test_pred = svm.classify(test_data)
    print('test accuracy = {}'.format((test_pred == test_targets).mean()))

    w_T = svm.w[1:].reshape((28,28))
    plt.imshow(w_T, cmap='gray')
    plt.title("w as a 28*28 image")
    plt.show()






