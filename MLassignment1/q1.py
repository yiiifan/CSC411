from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np



def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    # Show the deminsion of data and targets.
    print('number of data points: ', boston.data.size)
    print('Demensions : ', boston.data.shape)
    print('Target: ', boston.target.shape)
    return X, y, features


def visualize(X, y, features):
    figure1 = plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        fig = plt.scatter(X[:,i], y, s=1)
        plt.title(features[i])
        plt.xlabel('data_point')
        plt.ylabel('target')

    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    # raise NotImplementedError()
    XT = X .T
    A = np.dot(XT, X)
    B = np.dot(XT, Y)
    w = np.linalg.solve(A, B)
    return w


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # Add a bias to model
    column1 = np.ones((len(X), 1))
    X = np.concatenate((column1, X), axis=1)

    # TODO: Split data into train and test
    s = len(X)
    randomlist = np.random.permutation(s)
    k = int(0.8*s)
    trainingset = randomlist[0:k]
    testset = randomlist[k+1:s]

    X_train = X[trainingset]
    Y_train = y[trainingset]
    X_test = X[testset]
    Y_test = y[testset]

    # Fit regression model
    w = fit_regression(X_train, Y_train)
    print('w: ', w)


    # Plot feature y_prediction against y
    Y_predict = np.dot(X_test, w)
    plt.scatter(Y_test, Y_predict, s=1)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
    plt.show()
    print(Y_predict.shape, X_test.shape, w.shape)

    # choice the most significant features
    X_var = np.var(X_test[:,1])
    Y_var = np.var(Y_predict)
    loop = 10
    c = np.zeros((loop, 14))

    for i in range(14):
        corr = np.corrcoef(X_test[:, i], Y_predict)
        print(corr[1, 0])









    # Compute fitted values, MSE, etc.
    mse = np.mean(((Y_test - Y_predict)**2))
    print('MSE is :', mse)

    mae = np.mean(abs(Y_test - Y_predict))
    print('MAE is :', mae)

    rmse = np.sqrt(mse)
    print('RMSE is :', rmse)



if __name__ == "__main__":
    main()


