'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    print(train_data.shape, train_labels.shape)
    eta = np.zeros((10, 64))

    X = np.zeros((10, 700, 64))
    for i in range(10):
        X[i] = data.get_digits_by_label(train_data, train_labels, i)

        for m in range(64):
            cnt = 0
            for n in range(700):
                if X[i,n,m] == 1:
                    cnt = cnt + 1
            eta[i,m] = cnt/700
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    img = np.zeros((10,8,8))
    for i in range(10):
        img[i] = class_images[i].reshape((8, 8))

    all_concat = np.concatenate(img, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    img = np.zeros((10, 8, 8))
    for i in range(10):
        for j in range(64):
            n = 1
            p = eta[i][j]
            generated_data[i][j] = np.mean(np.random.binomial(n, p, 50))

        img[i] = generated_data[i].reshape((8,8))

    all_concat = np.concatenate(img, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    n =len(bin_digits)
    gen = np.zeros((n,10))
    for i in range(n):
        for k in range(10):
            for j in range(64):
                gen[i,k] = gen[i,k] + (bin_digits[i][j]*np.log(eta[k][j])+(1-bin_digits[i][j])*np.log(1-(eta[k][j])))

    return gen

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = len(bin_digits)
    con = np.zeros((n,10))
    gen = generative_likelihood(bin_digits, eta)
    for i in range(n):
        de = 0
        for j in range(10):
            de = de + np.exp(gen[i][j])*0.1
        con[i] = gen[i] + np.log(0.1) - np.log(de)

    return con


def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    n = labels.size

    aver = np.zeros((10,))
    cnt = np.zeros((10,))
    for i in range(n):
        label = int(labels[i])
        aver[label] = aver[label] + cond_likelihood[i, label]
        cnt[label] = cnt[label] + 1

    for j in range(10):
        aver[j] = aver[j] / cnt[j]

    print('aver is :', aver)

    # Compute as described above and return
    return aver

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    n = 4000
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    labels = np.zeros((n,))
    for i in range(n):
        labels[i] = np.argmax(cond_likelihood[i])
        #print(i)
    return labels

def accurancy(digits, labels, eta):

    lab = classify_data(digits, eta)
    n = len(lab)
    temp = 0
    for i in range(n):
        if lab[i] == labels[i]:
            temp = temp + 1

    accur = temp / n
    return accur

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)

    train_avg = avg_conditional_likelihood(train_data, train_labels, eta)
    test_avg = avg_conditional_likelihood(test_data, test_labels, eta)

    train_accur = accurancy(train_data, train_labels,eta)
    test_accur = accurancy(test_data, test_labels,eta)
    print('The train accuracy is %f, the test accuracy is %f.'%(train_accur,test_accur))

if __name__ == '__main__':
    main()
