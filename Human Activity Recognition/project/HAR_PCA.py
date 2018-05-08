import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import HAR_prediction


def plotGraph3D(train,fx,fy,fz):
    div = train.groupby('Activity')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for name,group in div:
        ax.scatter(group[fx],group[fy],group[fz],'.',label=name)
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_zlabel(fz)
    ax.legend()
    plt.show()

def perform_PCA(X_data, y_data):

        # perform PCA
        num=300
        variance=np.zeros((num,1))
        for i in range(1,num):
            pca = PCA(n_components=i)
            X_trained = pca.fit_transform(X_data, y=y_data)
            variance[i] = pca.explained_variance_ratio_.sum()
            print(i,variance[i])
        x = range(1,num+1)
        plt.plot(x, variance.reshape(num,1))
        plt.xlabel("No. of components")
        plt.ylabel("sum of variance ratio")
        plt.title("Sum of variance ratio w.r.t no. of components")
        plt.show()

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_data)

        # prepare the data for seaborn
        print('Creating plot for this PCA visualization..')
        df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': y_data})

        # draw the plot in appropriate place in the grid
        sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8, \
                   palette="Set1", markers=['^', 'v', 's', 'o', '1', '2'])
        #plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
        print('saving this plot as image in present working directory...')
        plt.show()
        print('Done')

def try_PCA(X_train, X_test,n_components):

    pca = PCA(n_components)  # choose num components
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    # how much info is compressed into the first few components
    print(pca.explained_variance_ratio_.shape)
    # cumulative variance(figure out how many components to keep at least 70% )
    # value = 1 means 100 of dataset's info is captured the components shown that were returned(we don't want that as it contain noise, redundancy and outliers)
    print(pca.explained_variance_ratio_.sum())

    return X_t_train,X_t_test

def try_PCAkernel(X_train, X_test,n_components):

    pca = KernelPCA(n_components, kernel='rbf', gamma=0.1)
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)

    return X_t_train, X_t_test

def perform_PCAkernel(X_train, X_labels,n_components):
    pca = KernelPCA(n_components)
    train_pca = pca.fit(X_train, X_labels)
    y = pca.lambdas_
    x = range(1, n_components+1)
    plt.plot(x, y)
    plt.xlabel("No. of components")
    plt.ylabel("Eigen values")
    plt.title("Data preserved w.r.t no. of components")
    plt.show()

    pca = KernelPCA(n_components=2,kernel='rbf', gamma=0.1)
    X_reduced = pca.fit_transform(X_train)

    # prepare the data for seaborn
    print('Creating plot for this KernelPCA visualization..')
    df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'label': X_labels})

    # draw the plot in appropriate place in the grid
    sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8, \
               palette="Set1", markers=['^', 'v', 's', 'o', '1', '2'])
    plt.show()
    print('Done')

def svm_PCA(X_train, X_test, y_train, y_test):
    accuTrain = []
    accuTest = []
    num=200
    for i in range(1,num):
        pca = PCA(n_components=i)
        train_pca = pca.fit_transform(X_train,y=y_train)
        print(i,pca.explained_variance_ratio_.sum())
        test_pca = pca.transform(X_test)
        clf = svm.LinearSVC(C=0.5)
        clf.fit(train_pca,y_train)
        accuTest.append(clf.score(test_pca,y_test))
        accuTrain.append(clf.score(train_pca,y_train))
    comp = [i for i in range(1,num)]
    plt.plot(comp,accuTrain,label="Train Accuracy")
    plt.plot(comp,accuTest,label="Test Accuracy")
    plt.xlabel("No. of Components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy V/S Components")
    plt.legend(loc='best')
    plt.show()

def lr_PCA(X_train, X_test, y_train, y_test):
        accuTrain1 = []
        accuTest1 = []
        accuTrain2 = []
        accuTest2 = []
        num = 50
        for i in range(1, num):
            pca = PCA(n_components=i)
            train_pca = pca.fit_transform(X_train, y=y_train)
            print(i, pca.explained_variance_ratio_.sum())
            test_pca = pca.transform(X_test)

            kpca = KernelPCA(n_components=i, kernel='rbf', gamma=0.1)
            kpca.fit(X_train)
            train_kpca = kpca.transform(X_train)
            test_kpca = kpca.transform(X_test)



            clf = linear_model.LogisticRegression(C=20, penalty='l1')
            clf.fit(train_pca, y_train)
            accuTest1.append(clf.score(test_pca, y_test))
            accuTrain1.append(clf.score(train_pca, y_train))

            clf.fit(train_kpca, y_train)
            accuTest2.append(clf.score(test_kpca, y_test))
            accuTrain2.append(clf.score(train_kpca, y_train))
        comp = [i for i in range(1, num)]
        plt.plot(comp, accuTrain1, label="PCA Train Accuracy")
        plt.plot(comp, accuTest1, label="PCA Test Accuracy")
        plt.plot(comp, accuTrain2, label="KernelPCA Train Accuracy")
        plt.plot(comp, accuTest2, label="KernelPCA Test Accuracy")
        plt.xlabel("No. of Components")
        plt.ylabel("Accuracy")
        plt.title("Accuracy V/S Components")
        plt.legend(loc='best')
        plt.show()

        print('PCA Train Accuracy')
        print(accuTrain1)
        print("PCA Test Accuracy")
        print(accuTest1)
        print("KernelPCA Train Accuracy")
        print(accuTrain2)
        print("KernelPCA Test Accuracy")
        print(accuTest2)


def svmk_PCA(X_train, X_test, y_train, y_test):
    accuTrain = []
    accuTest = []
    num = 200
    for i in range(1, num):
        pca = PCA(n_components=i)
        train_pca = pca.fit_transform(X_train, y=y_train)
        print(i, pca.explained_variance_ratio_.sum())
        test_pca = pca.transform(X_test)
        clf = SVC(kernel='rbf', C=16, gamma=0.0078125)
        clf.fit(train_pca, y_train)
        accuTest.append(clf.score(test_pca, y_test))
        accuTrain.append(clf.score(train_pca, y_train))
    comp = [i for i in range(1, num)]
    plt.plot(comp, accuTrain, label="Train Accuracy")
    plt.plot(comp, accuTest, label="Test Accuracy")
    plt.xlabel("No. of Components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy V/S Components")
    plt.legend(loc='best')
    plt.show()

def dt_PCA(X_train, X_test, y_train, y_test):
        accuTrain = []
        accuTest = []
        num = 200
        for i in range(1, num):
            pca = PCA(n_components=i)
            train_pca = pca.fit_transform(X_train, y=y_train)
            print(i, pca.explained_variance_ratio_.sum())
            test_pca = pca.transform(X_test)
            clf = DecisionTreeClassifier(max_depth=7)
            clf.fit(train_pca, y_train)
            accuTest.append(clf.score(test_pca, y_test))
            accuTrain.append(clf.score(train_pca, y_train))
        comp = [i for i in range(1, num)]
        plt.plot(comp, accuTrain, label="Train Accuracy")
        plt.plot(comp, accuTest, label="Test Accuracy")
        plt.xlabel("No. of Components")
        plt.ylabel("Accuracy")
        plt.title("Accuracy V/S Components")
        plt.legend(loc='best')
        plt.show()

def knn_PCA(X_train, X_test, y_train, y_test):
        accuTrain = []
        accuTest = []
        num = 200
        for i in range(1, num):
            pca = PCA(n_components=i)
            train_pca = pca.fit_transform(X_train, y=y_train)
            print(i, pca.explained_variance_ratio_.sum())
            test_pca = pca.transform(X_test)
            clf = KNeighborsClassifier(n_neighbors=9)
            clf.fit(train_pca, y_train)
            accuTest.append(clf.score(test_pca, y_test))
            accuTrain.append(clf.score(train_pca, y_train))
        comp = [i for i in range(1, num)]
        plt.plot(comp, accuTrain, label="Train Accuracy")
        plt.plot(comp, accuTest, label="Test Accuracy")
        plt.xlabel("No. of Components")
        plt.ylabel("Accuracy")
        plt.title("Accuracy V/S Components")
        plt.legend(loc='best')
        plt.show()

def main():
    train = pd.read_csv('UCI_HAR_dataset/csv_files/train.csv')
    test = pd.read_csv('UCI_HAR_dataset/csv_files/test.csv')
    print(train.shape, test.shape)

    X_pre_pca = train.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_train = train['ActivityName']
    X_test_pre_pca = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
    y_test = test['ActivityName']
    print(X_pre_pca.shape, y_train.shape)
    print(X_test_pre_pca.shape, y_test.shape)
    #perform_PCA(X_data=X_pre_pca, y_data=y_train)
    #perform_PCAkernel(X_pre_pca,y_train,150)

    #X_t_train, X_t_test = try_PCA(X_pre_pca, X_test_pre_pca, 150)
    #print(X_t_train.shape, X_t_test.shape)

    #X_t_train, X_t_test = try_PCAkernel(X_pre_pca, X_test_pre_pca, 20)
    #print(X_t_train.shape, X_t_test.shape)

    #HAR_prediction.Log_regression(X_t_train, X_t_test,y_train, y_test)
    #HAR_prediction.SVM(X_t_train, X_t_test,y_train, y_test)
    #HAR_prediction.SVM_kernel(X_t_train, X_t_test,y_train, y_test)
    #HAR_prediction.DecisionTree(X_t_train, X_t_test,y_train, y_test)
    #HAR_prediction.KNN(X_t_train, X_t_test,y_train, y_test)

    lr_PCA(X_pre_pca, X_test_pre_pca,y_train, y_test)
    #svm_PCA(X_pre_pca, X_test_pre_pca,y_train, y_test)
    #svmk_PCA(X_pre_pca, X_test_pre_pca,y_train, y_test)
    #knn_PCA(X_pre_pca, X_test_pre_pca,y_train, y_test)



if __name__ == '__main__':
    main()




