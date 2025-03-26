import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

N = 200
#make_classification() returns two numpy arrays -> X, y
X, y = make_classification(n_samples = N, n_features = 2, n_classes = 2, n_clusters_per_class = 1, random_state = 42, n_redundant = 0)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'rainbow')#c=y means each class will have a different color based on the value of y
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = 'rainbow')#c=y means each class will have a different color based on the value of y
# plt.show()


l_rate = 0.5
def y_hat(x,w,b):
    return sigmoid(np.dot(x,w) + b)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def djw(x,y,w,b):
    # print(x.shape)
    # print(y.shape)
    # print(f"x shape: {x.shape}, y shape: {y.shape}, w shape: {w.shape}")
    if x.ndim == 1:  # Single sample
        return -x * (y - y_hat(x, w, b))
    else:  # Batch
        return -(1 / len(y)) * np.dot(x.T, (y - y_hat(x, w, b)))
    # diff = y - y_hat(x,w,b)
    # sum = np.dot(x.T, diff)
    # return (-1/len(y)) * sum
def djb(x,y,w,b):
    # diff = y - y_hat(x, w, b)
    # sum = np.dot(x.T, diff)
    # return (-1 / len(y)) * sum
     if x.ndim == 1:  # Single sample
        return -(y - y_hat(x, w, b))
     else:  # Batch
        return -(1/len(y)) * np.sum(y - y_hat(x, w, b))

def plot_db(X, y, w, b, s):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xp, yp = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    F = sigmoid(w[0]*xp + w[1]*yp + b)
    plt.contour(xp, yp, F, levels=[0.5], cmap = 'rainbow')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'rainbow')
    plt.title(s)
    plt.show()

def batch_gradient_descent(X_train, y_train, learning_rate, num_iterations, w, b):
    i = 0
    while i < num_iterations:
        w = w - learning_rate * djw(X_train, y_train, w, b)
        b = b - learning_rate * djb(X_train, y_train, w, b)
        i+=1
    return w, b
def stochastic_gradient_descent(X_train, y_train, learning_rate, num_iterations, w, b):
    i = 0
    while i < num_iterations:
        rand_int = np.random.randint(low=0, high=len(X_train))
        w = w - learning_rate * djw(X_train[rand_int,:], y_train[rand_int], w, b)
        b = b - learning_rate * djb(X_train[rand_int,:], y_train[rand_int], w, b)
        i+=1
    return w, b
def minibatch_gradient_descent(X_train, y_train, learning_rate, num_iterations, w, b):
    i = 0
    while i < num_iterations:
        w = w - learning_rate * djw(X_train[50:100,:], y_train[50:100], w, b)
        b = b - learning_rate * djb(X_train[50:100,:], y_train[50:100], w, b)
        i+=1
    return w, b

############
def predict(x,w,b):
    predict = sigmoid(np.dot(x,w)+b)
    return 0 if predict < 0.5 else 1

init_w = np.zeros(X_train.shape[1])
init_b = 0
num_iterations = 2000
algo = {0:'BGD',1:'SGD',2:'MGBD'}
correct_BGD = 0
correct_SGD = 0
correct_MBGD = 0
w_BGD,b_BGD = batch_gradient_descent(X_train, y_train, learning_rate = l_rate, num_iterations = num_iterations, w=init_w, b=init_b)
w_SGD,b_SGD = stochastic_gradient_descent(X_train, y_train, learning_rate = l_rate, num_iterations = num_iterations, w=init_w, b=init_b)
w_MBGD,b_MBGD = minibatch_gradient_descent(X_train, y_train, learning_rate = l_rate, num_iterations = num_iterations, w=init_w, b=init_b)
for i in range(len(X_test)):
    y_predict_BGD = predict(X_test[i], w_BGD, b_BGD)
    y_predict_SGD = predict(X_test[i], w_SGD, b_SGD)
    y_predict_MBGD = predict(X_test[i], w_MBGD, b_MBGD)
    if y_predict_BGD == y_test[i]:
        correct_BGD += 1
    if y_predict_SGD == y_test[i]:
        correct_SGD += 1
    if y_predict_MBGD == y_test[i]:
        correct_MBGD += 1
BGD_accuracy = (correct_BGD/len(X_test))*100
SGD_accuracy = (correct_SGD/len(X_test))*100
MBGD_accuracy = (correct_MBGD/len(X_test))*100
print(f"BGD accuracy: {BGD_accuracy}")
print(f"SGD accuracy: {SGD_accuracy}")
print(f"MBGD accuracy: {MBGD_accuracy}")
plot_db(X_train, y_train, w_BGD, b_BGD,"BGD Decision Boundary")
plot_db(X_train, y_train, w_SGD, b_SGD, "SGD Decision Boundary")
plot_db(X_train, y_train, w_MBGD, b_MBGD, "MBGD Decision Boundary")
##################
#scikit-learn logistic regression
def plot_db_sl(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict on grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) #flatten and c_ stack column wise
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolors='face') #edge same color as points
    plt.title("Scikit-Learn Logistic Regression Decision Boundary")
    plt.show()

def train_logistic():
    model = LogisticRegression()
    model.fit(X_train, y_train)
    plot_db_sl(X_train, y_train, model)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Scikit Learn Accuracy: {accuracy*100:.2f}")

train_logistic()
