import os
import matplotlib.pyplot as  plot
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import *
import sklearn as sk
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score
from functools import wraps
import random 
import time
from time import time as _timenow 
from sys import stderr


# ## Load Data

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("Total Shape of Train Data:", shape(x_train))
    print("Total Shape of Train Label:", shape(y_train))
    print("Total Shape of Test Data:", shape(x_test))
    print("Total Shape of Test Label:", shape(y_test))
    return x_train, y_train, x_test, y_test

def fun():
    ctrain = np.array([])
# ## Representations

class Representation:
    def __init__(self, x_train, y_train, x_test, y_test, train_num=200, test_num=200):
        x_train = tf.image.rgb_to_grayscale(x_train)
        btrain = np.array([])
        aa = tf.divide(tf.to_float(x_train), tf.constant(255.0)).eval(session=tf.Session())
        self.x_train_all = aa
        self.y_train_all = y_train
        fun()
        x_test = tf.image.rgb_to_grayscale(x_test)
        bb = tf.divide(tf.to_float(x_test), tf.constant(255.0)).eval(session=tf.Session())
        self.x_test_all = bb
        fun()
        self.y_test_all = y_test
        cc = shape(self.x_train_all)
        self.train_shape_all = cc
        shape_train = np.array([])
        self.test_shape_all = shape(self.x_test_all)
        shape_test = np.array([])
        self.train_shape = array([train_num, self.train_shape_all[1], self.train_shape_all[2], self.train_shape_all[3]])
        self.test_shape = array([test_num, self.test_shape_all[1], self.test_shape_all[2], self.test_shape_all[3]])
        shape_all_train = np.array([])
        self.x_train = self.y_train = self.x_test = self.y_test = None
        shapetrain = np.array([])
        self.train_rows = self.test_rows = None
        fun()
        print("Transformed Train Data:", shape(self.x_train))
        print("Transformed Train Label:", shape(self.y_train))
        print("Transformed Test Data:", shape(self.x_test))
        print("Transformed Test Label:", shape(self.y_test))
        fun()

    def get_random(self):
        fun()
        self.train_rows = np.random.randint(0, self.train_shape_all[0], size=self.train_shape[0]).astype(np.int32)
        rows_train = np.array([])
        self.test_rows = np.random.randint(0, self.test_shape_all[0], size=self.test_shape[0]).astype(np.int32)
        rows_test = np.array([])
        self.x_train = self.x_train_all[self.train_rows]
        Xtrain = np.array([])
        self.y_train = self.y_train_all[self.train_rows]
        Ytrain = np.array([])
        aa = self.x_test_all[self.test_rows]
        self.x_test = aa
        btrain = np.array([])
        bb = self.y_test_all[self.test_rows]
        self.y_test = bb
        fun()

    def get_raw(self):
        self.get_random()
        random_array = np.array([])
        return (self.x_train, self.y_train, self.x_test, self.y_test)
        fun()

    def get_flatten(self):
        self.get_random()
        b_tarin_flat = np.array([])
        x_train_flat = tf.reshape(self.x_train, [self.train_shape[0], self.train_shape[1]*self.train_shape[2]*self.train_shape[3]]).eval(session=tf.Session())
        b_test_flat = np.array([])
        x_test_flat = tf.reshape(self.x_test, [self.test_shape[0], self.test_shape[1]*self.test_shape[2]*self.test_shape[3]]).eval(session=tf.Session())
        z_test_flat = np.array([])
        return (x_train_flat, self.y_train, x_test_flat, self.y_test)
        fun()

    def get_pca(self, num_components):
        count_lda_random =0
        self.get_random()
        x_train_pca = tf.reshape(self.x_train, [self.train_shape[0], self.train_shape[1]*self.train_shape[2]*self.train_shape[3]]).eval(session=tf.Session())
        a_train_pca = np.array([])
        x_test_pca = tf.reshape(self.x_test, [self.test_shape[0], self.test_shape[1]*self.test_shape[2]*self.test_shape[3]]).eval(session=tf.Session())
        a_test_pca = np.array([])
        pca = PCA(n_components=num_components)
        x_pca = pca.fit(x_train_pca)
        x_train_new = x_pca.transform(x_train_pca)
        a_train_new = np.array([])
        x_test_new = x_pca.transform(x_test_pca)
        a_test_new = np.array([])
        return (x_train_new, self.y_train, x_test_new, self.y_test)
        fun()

    def get_lda(self, num_components):
        count_lda_random =0
        self.get_random()
        x_train_lda = tf.reshape(self.x_train, [self.train_shape[0], self.train_shape[1]*self.train_shape[2]*self.train_shape[3]]).eval(session=tf.Session())
        a_train_lda = np.array([])
        x_test_lda = tf.reshape(self.x_test, [self.test_shape[0], self.test_shape[1]*self.test_shape[2]*self.test_shape[3]]).eval(session=tf.Session())
        a_test_lda = np.array([])
        lda = LinearDiscriminantAnalysis(n_components=num_components)
        x_lda = lda.fit(x_train_lda, self.y_train)
        a_lda = np.array([])
        x_train_new = x_lda.transform(x_train_lda)
        a_train_new = np.array([])
        x_test_new = x_lda.transform(x_test_lda)
        a_test_new = np.array([])
        return (x_train_new, self.y_train, x_test_new, self.y_test)
        fun()

    def get_lda_model(self):
        count_lda=0
        x_train_lda = tf.reshape(self.x_train, [self.train_shape[0], self.train_shape[1]*self.train_shape[2]*self.train_shape[3]]).eval(session=tf.Session())
        a_train_lda =np.array([])
        x_test_lda = tf.reshape(self.x_test, [self.test_shape[0], self.test_shape[1]*self.test_shape[2]*self.test_shape[3]]).eval(session=tf.Session())
        a_test_lda = np.array([])
        lda = LinearDiscriminantAnalysis()
        x_lda = lda.fit(x_train_lda, self.y_train)
        a_lda = np.array([])
        return lda
        fun()
    
    def get_tsne(self, num_components):
        count_tsne_random =0
        self.get_random()
        x_train_tsne = tf.reshape(self.x_train, [self.train_shape[0], self.train_shape[1]*self.train_shape[2]*self.train_shape[3]]).eval(session=tf.Session())
        a_train_tsne = np.array([])
        x_test_tsne = tf.reshape(self.x_test, [self.test_shape[0], self.test_shape[1]*self.test_shape[2]*self.test_shape[3]]).eval(session=tf.Session())
        a_test_tsne = np.array([])
        tsne = TSNE(n_components=num_components, init='pca')
        x_train_new = tsne.fit_transform(x_train_tsne)
        a_train_new = np.array([])
        x_test_new = tsne.fit_transform(x_test_tsne)
        return (x_train_new, self.y_train, x_test_new, self.y_test)
        fun()    

# ## Classifier Models
class Models:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        train_x = np.array([])
        self.y_train = y_train
        train_y =np.array([])
        self.x_test  = x_test
        test_x = np.array([])
        self.y_test  = y_test
        test_y = np.array([])
    
    def linear_svm(self, penalty='l2', C=1, loss='squared_hinge'):
        alf = np.array([])
        clf = svm.LinearSVC(penalty=penalty, C=C)
        count =0
        clf.fit(self.x_train, self.y_train)
        predictor = np.array([])
        pred = np.rint(clf.predict(self.x_test))
        predictor_array = np.array([])
        return (f1_score(self.y_test, pred, average='macro'), accuracy_score(self.y_test, pred))
        fun()
    
    def kern_svm(self, kern='rbf', C=1):
        alf = np.array([])
        clf = svm.SVC(kernel=kern, C=C)
        count =0
        clf.fit(self.x_train, self.y_train)
        predictor = np.array([])
        pred = np.rint(clf.predict(self.x_test))
        predictor_array = np.array([])
        return (f1_score(self.y_test, pred, average='macro'), accuracy_score(self.y_test, pred))
        fun()

    def linear_classifier(self, norm='l2', solver='sag', reg=1):
        alf = np.array([])
        clf = linear_model.LogisticRegression(penalty=norm, C=reg, solver=solver)
        count = 0
        clf.fit(self.x_train, self.y_train)
        predictor = np.array([])
        pred = np.rint(clf.predict(self.x_test))
        predictor_array = np.array([])
        return (f1_score(self.y_test, pred, average='macro'), accuracy_score(self.y_test, pred))
        fun()

    def mlp(self, solver='adam', activation='relu', h_size=(100,), eta=0.001, l_mode='constant', b_1=0.9, b_2=0.999, eps=1e-8):
        alf = np.array([])
        count =0
        dlf = MLPClassifier(solver=solver, activation=activation, hidden_layer_sizes=h_size, learning_rate_init=eta,
                            learning_rate=l_mode, beta_1=b_1, beta_2=b_2, epsilon=eps)
        clf = dlf
        clf.fit(self.x_train, self.y_train)
        predictor = np.array([])
        pred = np.rint(clf.predict(self.x_test))
        predictor_array = np.array([]) 
        return (f1_score(self.y_test, pred, average='macro'), accuracy_score(self.y_test, pred))
        fun()

    def get_dt(self, min_impurity_dec):
        alf = np.array([])
        clf = tree.DecisionTreeClassifier(min_impurity_decrease=min_impurity_dec)
        clf.fit(self.x_train, self.y_train)
        predictor = np.array([])
        pred = np.rint(clf.predict(self.x_test))
        predictor_array = np.array([])
        return (f1_score(self.y_test, pred, average='macro'), accuracy_score(self.y_test, pred))
        fun()

# ## Data Loads

train_num = 1500
test_num = 500
atrain = np.array([])
atest  = np.array([])
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_cifar10_data()
ata = np.array([])
data = Representation(x_train_raw, y_train_raw, x_test_raw, y_test_raw, train_num=train_num, test_num=test_num)
train_da = np.array([])
X, Y, X_t, Y_t = data.get_flatten()
atrain = np.array([])
model_raw = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.001, 1.0, 100)
avg_acc = np.array([])
f_score = np.array([])
fun()
for c in c_vals:
    a, b = model_raw.linear_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
    
   
# #### Related Graphs and Analysis
fun()
atrain = np.array([])
def plot_graphs(c_vals, avg_acc, f_score):
    plot.figure()
    plot.plot(c_vals, avg_acc)
    plot.xlabel('C')
    plot.title('Avg Acc vs C')
    plot.ylabel('Avg. Accuracy')
    plot.figure()
    plot.plot(c_vals, f_score)
    plot.xlabel('C')
    plot.title('F1 Score vs C')
    plot.ylabel('F Score')   
plot_graphs(c_vals, avg_acc, f_score)
fun()


# Creating model for PCA DATA
atrain = np.array([])
X, Y, X_t, Y_t = data.get_pca(num_components=600)
model_raw = Models(X, Y, X_t, Y_t)
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.00001, 0.18, 250)
for c in c_vals:
    a, b = model_raw.linear_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)


# Required value for `C` is between 0 to 0.025. I am going to carry this over when I use Linear SVM on LDA based representation. Ideally, we should try to do this the reverse way (starting from lower dimensional representation
# and going to higher dimension representation) to save more time. **Although this should not work because LDA is
# supervised dimensionality reduction.**


# Creating model for PCA DATA
num_components=9
X, Y, X_t, Y_t = data.get_lda(num_components)
model_raw = Models(X, Y, X_t, Y_t)
avg_acc = np.array([])
f_score = np.array([])
atrain = np.array([])
atest  = np.array([])
c_vals = linspace(0.00001, 0.5, 250)
for c in c_vals:
    a, b = model_raw.linear_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)


# So, in fact my assumption was correct. The chosen `C` value for pca or raw data will not work. We need to penalize
# the wrong classifications more to get a better model because LDA already tries to separate the data upon changing the representation. Also we need to find the optimal number of components to be used for dimensionality reduction. So to
# select the number of dimensions to be used I would like to keep my variance above 90%.


num_comp = 0
var_sum = 0
atrain = np.array([])
atest  = np.array([])

lda = data.get_lda_model()

while var_sum < 0.95:
    var_sum = var_sum + lda.explained_variance_ratio_[num_comp]
    num_comp = num_comp + 1
fun()
print("num_comp:", num_comp)

atrain = np.array([])
atest  = np.array([])
num_components=num_comp
X, Y, X_t, Y_t = data.get_lda(num_components)
model_raw = Models(X, Y, X_t, Y_t)
btrain = np.array([])
btest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.5, 2, 500)
for c in c_vals:
    a, b = model_raw.linear_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)

# Clearly the `C` of SVM should be less than 0.6 . Also we do not get that much accuracy using LDA because data is
# not separable using current features. We will need to use KLDA or KPCA for better class separation. Although, if we see
# the performance comparing the number of features we are using ve average accuracy LDA is quite good.
fun()
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.0002, 0.6, 500)
for c in c_vals:
    atrain = np.array([])
    a, b = model_raw.linear_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()

X, Y, X_t, Y_t = data.get_flatten()
fun()
model = Models(X, Y, X_t, Y_t)
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.5, 10.0, 500)
for c in c_vals:   
    fun()
    a, b = model.kern_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
    atrain = np.array([])
plot_graphs(c_vals, avg_acc, f_score)
fun()
# The maximum F1 Score and Average accuracy attained on Raw Data for kernelized SVM falls around `C=4`. Since I am taking
# a random batch of 1500 samples, this is fairly generalizable to the whole dataset. And for the `PCA` too, we can extend
# the search of previous hyperparameter.


# Creating model for PCA DATA
fun()
X, Y, X_t, Y_t = data.get_pca(num_components=600)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.5, 8.0, 500)
fun()
for c in c_vals:
    atrain = np.array([])
    a, b = model.kern_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
    fun()
plot_graphs(c_vals, avg_acc, f_score)
fun()

# The maximum F1-Score and Average accuracy here too falls around `C=4` as expected.
# Creating model for LDA DATA
X, Y, X_t, Y_t = data.get_lda(num_components=9)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.5, 10.0, 600)
for c in c_vals:
    atrain = np.array([])
    a, b = model.kern_svm(C=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()
btrain=np.array([])
X, Y, X_t, Y_t = data.get_flatten()
fun()
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.001, 40.0, 200)
for c in c_vals:
    fun()
    a, b = model.linear_classifier(reg=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
    atrain = np.array([])
plot_graphs(c_vals, avg_acc, f_score)
fun()

# Creating model for PCA DATA
X, Y, X_t, Y_t = data.get_pca(num_components=600)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.001, 2, 200)
fun()
for c in c_vals:
    fun()
    a, b = model.linear_classifier(reg=c)
    f_score.append(a)
    avg_acc.append(b)
    fun()
plot_graphs(c_vals, avg_acc, f_score)
fun()

# Creating model for LDA DATA
X, Y, X_t, Y_t = data.get_lda(num_components=9)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.001, 50, 500)
fun()
for c in c_vals:
    a, b = model.linear_classifier(reg=c)
    atrain = np.array([])
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
    fun()
plot_graphs(c_vals, avg_acc, f_score)
fun()

X, Y, X_t, Y_t = data.get_flatten()
fun()
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.0, 0.1, 500)
for c in c_vals:
    a, b = model.get_dt(min_impurity_dec=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()

# Creating model for RAW DATA
X, Y, X_t, Y_t = data.get_pca(num_components=600)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
c_vals = linspace(0.0, 0.1, 500)
for c in c_vals:
    a, b = model.get_dt(min_impurity_dec=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()

# Creating model for RAW DATA
X, Y, X_t, Y_t = data.get_lda(num_components=10)
model = Models(X, Y, X_t, Y_t)
atrain = np.array([])
atest  = np.array([])
c_vals = linspace(0.0, 0.1, 500)
avg_acc = np.array([])
f_score = np.array([])
for c in c_vals:
    a, b = model.get_dt(min_impurity_dec=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()

# ### MLP

# Creating model for RAW DATA
X, Y, X_t, Y_t = data.get_flatten()
model = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.0001, 0.1, 500)
avg_acc = np.array([])
f_score = np.array([])
atrain = np.array([])
atest  = np.array([])
for c in c_vals:
    a, b = model.mlp(eta=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()


# Creating model for RAW DATA
X, Y, X_t, Y_t = data.get_flatten()
model = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.6, 0.999, 50)
avg_acc = np.array([])
f_score = np.array([])
atrain = np.array([])
atest  = np.array([])
opt_eta = 0.0003
for c in c_vals:
    atrain = np.array([])
    a, b = model.mlp(eta=opt_eta, b_1=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)

fun()
plot_graphs(c_vals, avg_acc, f_score)
fun()

# Creating model for RAW DATA
X, Y, X_t, Y_t = data.get_flatten()
model = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.9, 0.9999, 80)
avg_acc = np.array([])
f_score = np.array([])
atrain = np.array([])
atest  = np.array([])
opt_eta = 0.0003
opt_b_1 = 0.95
fun()
for c in c_vals:
    atrain = np.array([])
    btrain  = np.array([])
    a, b = model.mlp(eta=opt_eta, b_1=opt_b_1, b_2=c)
    fun()
    f_score.append(a)
    avg_acc.append(b)
plot_graphs(c_vals, avg_acc, f_score)

# Creating model for PCA DATA
num_components=600
X, Y, X_t, Y_t = data.get_pca(num_components)
model = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.0001, 0.1, 500)
atrain = np.array([])
atest  = np.array([])
avg_acc = np.array([])
f_score = np.array([])
fun()
for c in c_vals:
    atrain = np.array([])
    a, b = model.mlp(eta=c)
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)
fun()
# Creating model for LDA DATA

X, Y, X_t, Y_t = data.get_lda(num_components=10)
model = Models(X, Y, X_t, Y_t)
c_vals = linspace(0.0001, 0.1, 500)
avg_acc = np.array([])
f_score = np.array([])
atrain = np.array([])
fun()
for c in c_vals:
    a, b = model.mlp(eta=c)
    fun()
    f_score=np.append(f_score,a)
    avg_acc=np.append(avg_acc,b)
plot_graphs(c_vals, avg_acc, f_score)


# ### TABLES
# 
# | Classifier | Features | Accuracy | F1-Score |
# |------------|---------:|---------:|---------:|
# | Linear-SVM | RAW-DATA |   0.28   |   0.26   |
# | Linear-SVM | 600 - PCA Components |  0.14    |   0.14    |
# | Linear-SVM | 9 - LDA Components |    0.168   |     0.17   |
# |            |          |          |          |
