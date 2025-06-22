import pyttb as ttb
import numpy as np



class Scaler:

    def __init__(self, X, mode, smin = 1.0e-12):
        pass

    def scale_tensor(self, X):
        return X.copy()
    
    def unscale_tensor(self, X, shift=True):
        return X.copy()
    
    def scale_ktensor(self, u, var=None):
        return u.copy()
    
    def unscale_ktensor(self, u, var=None):
        return u.copy()


class StdScaler(Scaler):

    def __init__(self, X, mode, smin = 1.0e-12):
        self.mode = mode

        # list of modes not including mode
        axis = list(range(X.ndims)) 
        axis.pop(self.mode)
        axis = tuple(axis)
    
        # commpute mean and standard deviation
        self.mu = np.mean(X.data, axis=axis)
        self.s = np.std(X.data, axis=axis)
        self.s[self.s<smin] = 1

        # compatible shape for mu,s for arithmetic with X
        shp = [1 for i in range(X.ndims)]
        shp[mode] = X.shape[mode]
        shp = tuple(shp)
        self.mt = np.reshape(self.mu,shp)
        self.st = np.reshape(self.s,shp)

        # compatible shape for s for arithmetic with factor matrices
        self.sk = np.reshape(self.s,(X.shape[mode],1))

    def scale_tensor(self, X):
        Xs = (X.data - self.mt) / self.st
        return ttb.tensor(Xs)
    
    def unscale_tensor(self, X, shift=True):
        if shift:
            Xs = X.data * self.st + self.mt
        else:
            Xs = X.data * self.st
        return ttb.tensor(Xs)
    
    def scale_ktensor(self, u, var=None):
        us = u.copy()
        if var is None:
            us.factor_matrices[self.mode] /= self.sk
        else:
            us.factor_matrices[self.mode][var,:] /= self.sk[var,:]
        return us
    
    def unscale_ktensor(self, u, var=None):
        us = u.copy()
        if var is None:
            us.factor_matrices[self.mode] *= self.sk
        else:
            us.factor_matrices[self.mode][var,:] *= self.sk[var,:]
        return us
