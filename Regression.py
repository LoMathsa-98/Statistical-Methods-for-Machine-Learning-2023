### RIDGE 
import numpy as np

class RidgeRegression:
    def __init__(self,reg_param = 1.0):
        self.reg_param = reg_param
        self.betas = None
        self.bias = None
        
    def fit(self, X, y, sample_weights=None):
        n,d = X.shape
        ones = np. ones ((n , 1))
        X = np. concatenate (( ones , X), axis =1)
        self.betas = np.zeros(d+1)
        I = np.eye(d+1)
        S = np.dot(X.T,X)
        self.betas = np.linalg.inv(S + self.reg_param * I).dot(X.T).dot(y)
        return self.betas
         
    def predict(self, X):
        n,d = X.shape
        ones = np. ones ((n , 1))
        X = np. concatenate (( ones , X), axis =1)
        return np.dot(X, self.betas)
    
    def get_params(self, deep=True):    
        return {'reg_param': self.reg_param}
    
    def get_coeff(self, deep=True):    
        return self.betas
    