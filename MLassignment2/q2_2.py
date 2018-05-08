'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    cnt = np.zeros(10)
    # Compute means
    for i in range(train_data.shape[0]):
        for j in range(10):
            if j == train_labels[i]:
                means[j] = means[j]+ train_data[i]
                cnt[j] = cnt[j]+1

    for t in range(10):
        means[t] = means[t] /cnt[t]

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    X = np.zeros((10,700,64))
    covariances = np.zeros((10,64,64))
    for i in range(10):
        X[i] = data.get_digits_by_label(train_data, train_labels, i)

    means = compute_mean_mles(train_data, train_labels)
    for m in range(10):
        for p in range(64):
            for q in range(64):
                covariances[m,p,q] = np.dot((X[m,:,p]-means[m,p]).T,(X[m,:,q]-means[m,q]))/700

    return covariances


def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10,8,8))
    for i in range(10):
        cov_diag[i] = np.log2(np.diag(covariances[i])).reshape((8, 8))

    all_concat = np.concatenate(cov_diag, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    n = digits.shape[0]
    pi = 3.1415926
    gen = np.zeros((n,10))
    for j in range(n):
        for i in range(10):
            d = 10
            co = covariances[i]
            mean = means[i]
            detco = np.linalg.det(co)
            invco = np.linalg.inv(co)
            distance = np.dot(np.dot((digits[j] - mean).T, invco), (digits[j] - mean))
            gen[j, i] = (-0.5) * (np.log(2 * pi) * d + np.log(detco) + distance)

    return gen

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = digits.shape[0]
    con = np.zeros((n,10))
    gen = generative_likelihood(digits, means, covariances)
    for i in range(n):
        de = 0
        for j in range(10):
            de = de + np.exp(gen[i][j])*0.1
        con[i] = gen[i] + np.log(0.1) - np.log(de)

    return con

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    n = labels.size
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    aver = np.zeros((10,))
    cnt = np.zeros((10,))
    for i in range(n):
        label = int(labels[i])
        aver[label] = aver[label] + cond_likelihood[i, label]
        cnt[label] = cnt[label]+1

    for j in range(10):
        aver[j] = aver[j]/cnt[j]
    print('aver is :', aver)

    # Compute as described above and return
    return aver


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    # Compute and return the most likely class
    n = 4000
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    labels = np.zeros((n,))
    for i in range(n):
        labels[i] = np.argmax(cond_likelihood[i][:])
        #print(i)
    return labels


def accurancy(digits, labels, means, covariances):

    lab = classify_data(digits, means, covariances)
    n = lab.size
    temp = 0
    for i in range(n):
        if lab[i] == labels[i]:
            temp = temp + 1

    accur = temp / n
    return accur



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # plot_cov_diagonal(covariances)
    plot_cov_diagonal(covariances)

    # Evaluation
    p = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    q = avg_conditional_likelihood(train_data, train_labels, means, covariances)


    train_accur = accurancy(train_data, train_labels, means, covariances)
    test_accur = accurancy(test_data, test_labels, means, covariances)

    print('The train accuracy is %f, the test accuracy is %f.'%(train_accur,test_accur))



if __name__ == '__main__':
    main()