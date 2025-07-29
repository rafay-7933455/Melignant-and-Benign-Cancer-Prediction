import numpy as np

def sigmoid(z):
     g = 1/(1+np.exp(-z))
     return g

def gradient_logistic(x, y, w: np.ndarray, b = 0):
     f_wb = sigmoid(np.dot(x, w)+b)
     m = x.shape[0]
     error = f_wb - y
     dj_dw = (1 / m) * np.dot(x.T, error)
     dj_db = (1 / m) * np.sum(error)
     return dj_dw, dj_db

def gradient_descent(x,y,w,b,n, iter):
     for i in range(iter):
          dj_dw, dj_db = gradient_logistic(x, y, w, b)
          w -= n * dj_dw
          b -= n * dj_db
     return w, b

def predict(x_test, w, b):
     return sigmoid(np.dot(x_test, w)+b)