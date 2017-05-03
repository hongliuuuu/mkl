from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,chi2_kernel,linear_kernel,polynomial_kernel,cosine_similarity
from sklearn import preprocessing
import xlrd
from sklearn.model_selection import GridSearchCV

def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
        indices.append(indice)
    train_indices = []
    for i in indices:
        k = int(len(i)*ratio)
        train_indices += (random.Random(seed).sample(i,k=k))
    #find the unused indices
    s = np.bincount(train_indices,minlength=n_samples)
    mask = s==0
    test_indices = np.arange(n_samples)[mask]
    return train_indices,test_indices
def Lsvm_patatune(train_x,train_y):
    tuned_parameters = [
        {'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['C']

def kn(X,Y):
    d = np.dot(X,Y)
    dx = np.sqrt(np.dot(X,X))
    dy = np.sqrt(np.dot(Y,Y))
    if(dx*dy==0):
        print(X,Y)
    k = pow((d/(dx*dy)+1),3)
    return k

def similarity(X):
    n_samples = X.shape[0]
    dis = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        dis[i][i] = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dis[i][j] = dis[j][i] = kn(X[i],X[j])
    return dis
def Lsvm_patatune(train_x,train_y,test_x, test_y):
    tuned_parameters = [
        {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100, 1000],'gamma': [0.0625, 0.125,0.25, 0.5, 1, 2, 5 ,7, 10, 12 ,15 ,17 ,20]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    print(clf.score(test_x,test_y))
    return clf.best_params_['C']
def gama_patatune(train_x,train_y,c):
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [0.0625, 0.125,0.25, 0.5, 1, 2, 5 ,7, 10, 12 ,15 ,17 ,20] }]
    clf = GridSearchCV(SVC(C=c), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['gamma']

url = 'Cal7_1.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]

for i in range(5):
    url = 'Cal7_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X1 = array[:, 1:]
    X = np.concatenate((X, X1), axis=1)
Y = pandas.read_csv('Cal7_label.csv')
Y = Y.values

Y = Y[:, 1:]
# Y = Y.transpose()
Y = np.ravel(Y)



#X = min_max_scaler.fit_transform(X)
#X1_features = similarity(X1)
#X2_features = similarity(X2)
#X3_features = similarity(X3)
"""


e1 = []
X1_features = polynomial_kernel(X1)+linear_kernel(X1)+rbf_kernel(X1)+laplacian_kernel(X1)
X2_features = linear_kernel(X2)+polynomial_kernel(X2)+rbf_kernel(X2)+laplacian_kernel(X2)
X3_features = linear_kernel(X3)+polynomial_kernel(X3)+rbf_kernel(X3)+laplacian_kernel(X3)
X_features = (X1_features + X2_features + X3_features)


for l in range(10):
    train_indices, test_indices = splitdata(X=X, Y=Y, ratio=0.7, seed=1000 + l)
    X_features1 = np.transpose(X_features)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    clf = SVC(kernel='precomputed')
    clf.fit(X_features3[train_indices], Y[train_indices])
    e1.append(clf.score(X_features3[test_indices], Y[test_indices]))
s = "combination of %d_%d_%d" % (l, l, l)
if np.mean(e1) > big:
    big = np.mean(e1)
    print(np.mean(e1))
    print(s)
testfile.write(s + ":%f" % (np.mean(e1)) + '\n')


"""
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Xnew1 = X[:, 0:48]
Xnew2 = X[:, 48:88]
Xnew3 = X[:, 88:342]
Xnew4 = X[:, 342:2326]
Xnew5 = X[:, 2326:2838]
Xnew6 = X[:, 2838:]
X_new = X
X = min_max_scaler.fit_transform(X)
X1 = X[:, 0:48]
X2 = X[:, 48:88]
X3 = X[:, 88:342]
X4 = X[:, 342:2326]
X5 = X[:, 2326:2838]
X6 = X[:, 2838:]
for r in range(3):
    if r==0:
        R = 0.3
    elif r==1:
        R = 0.5
    else:
        R = 0.7
    testfile = open("AAMKLcombinationTest%f.txt"%R,'w')
    big = 0
    mm = ""
    err = 0
    train_in, test_in = splitdata(X=X, Y=Y, ratio=R, seed=1000)
    trainx = X[train_in]
    trainy = Y[train_in]
    testx = X[test_in]
    testy = Y[test_in]
    c = Lsvm_patatune(train_x=trainx, train_y=trainy, test_x=testx, test_y=testy)
    print(c)
    g1 = gama_patatune(train_x=Xnew1[train_in], train_y=trainy, c=c)
    g2 = gama_patatune(train_x=Xnew2[train_in], train_y=trainy, c=c)
    g3 = gama_patatune(train_x=Xnew3[train_in], train_y=trainy, c=c)
    g4 = gama_patatune(train_x=Xnew4[train_in], train_y=trainy, c=c)
    g5 = gama_patatune(train_x=Xnew5[train_in], train_y=trainy, c=c)
    g6 = gama_patatune(train_x=Xnew6[train_in], train_y=trainy, c=c)
    print(g1, g2, g3, g4, g5)
    X1_features = rbf_kernel(Xnew1, gamma=g1)
    X2_features = rbf_kernel(Xnew2, gamma=g2)
    X3_features = rbf_kernel(Xnew3, gamma=g3)
    X4_features = rbf_kernel(Xnew4, gamma=g4)
    X5_features = rbf_kernel(Xnew5, gamma=g5)
    X6_features = rbf_kernel(Xnew6, gamma=g6)
    X_features = (X1_features + X2_features + X3_features + X4_features + X5_features+X6_features)

    e1 = []

    for l in range(10):
        train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=1000 + l)
        X_features1 = np.transpose(X_features)
        X_features2 = X_features1[train_indices]
        X_features3 = np.transpose(X_features2)

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features3[train_indices], Y[train_indices])
        e1.append(clf.score(X_features3[test_indices], Y[test_indices]))
    testfile.write(str(e1) + '\n')
    testfile.write(":%f \p %f" % (np.mean(e1), np.std(e1)) + '\n')
    """
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for o in range(4):
                    for p in range(4):
                        for q in range(4):
                            if(i==0):
                                X1_features = polynomial_kernel(X1)
                            elif(i==1):
                                X1_features = rbf_kernel(X1)
                            elif(i==2):
                                X1_features = laplacian_kernel(X1)
                            elif (i == 3):
                                X1_features = similarity(Xnew1)

                            if (j == 0):
                                X2_features = polynomial_kernel(X2)
                            elif (j == 1):
                                X2_features = rbf_kernel(X2)
                            elif (j == 2):
                                X2_features = laplacian_kernel(X2)
                            elif (j == 3):
                                X2_features = similarity(Xnew2)

                            if (k == 0):
                                X3_features = polynomial_kernel(X3)
                            elif (k == 1):
                                X3_features = rbf_kernel(X3)
                            elif (k == 2):
                                X3_features = laplacian_kernel(X3)
                            elif (k == 3):
                                X3_features = similarity(Xnew3)

                            if (o == 0):
                                X4_features = polynomial_kernel(X4)
                            elif (o == 1):
                                X4_features = rbf_kernel(X4)
                            elif (o == 2):
                                X4_features = laplacian_kernel(X4)
                            elif (o == 3):
                                X4_features = similarity(Xnew4)


                            if (p == 0):
                                X5_features = polynomial_kernel(X5)
                            elif (p == 1):
                                X5_features = rbf_kernel(X5)
                            elif (p == 2):
                                X5_features = laplacian_kernel(X5)
                            elif (p == 3):
                                X5_features = similarity(Xnew5)

                            if (q == 0):
                                X6_features = polynomial_kernel(X6)
                            elif (q == 1):
                                X6_features = rbf_kernel(X6)
                            elif (q == 2):
                                X6_features = laplacian_kernel(X6)
                            elif (q == 3):
                                X6_features = similarity(Xnew6)

                            X_features = (X1_features + X2_features + X3_features+X4_features+X5_features+X6_features)

                            e1 = []

                            for l in range(10):
                                train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=1000 + l)
                                X_features1 = np.transpose(X_features)
                                X_features2 = X_features1[train_indices]
                                X_features3 = np.transpose(X_features2)
                                c = Lsvm_patatune(train_x=X_features3[train_indices], train_y=Y[train_indices])
                                #print(c)
                                clf = SVC(C=c, kernel='precomputed')
                                clf.fit(X_features3[train_indices], Y[train_indices])
                                e1.append(clf.score(X_features3[test_indices], Y[test_indices]))
                            s = "combination of %d_%d_%d_%d_%d%d"%(i,j,k,o,p,q)
                            if np.mean(e1)>big:
                                big = np.mean(e1)
                                print(np.mean(e1))
                                print(s)
                                mm=s
                                err = big
                                std = np.std(e1)
                            testfile.write(s + ":%f \p %f" % (np.mean(e1), np.std(e1)) + '\n')

    testfile.write("best peformance is" + mm + ":%f \p %f" % (err, std) + '\n')
    """
    testfile.close()

