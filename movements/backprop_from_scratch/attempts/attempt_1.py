# Attempt 1 - Jun 10, 2026
# Started: 9:54pm
# First pass (loss drops, no NaN):
# Correct (acc > 90% on toy data):
# Stuck on:
# Notes:
# 
# t0 = x W0 + b0
# x1 = sigmoid(t0)
# t1 = x1 W1 + b1
# yp = sigmoid(t1)
# l = (y - yp)^2
# dl_dyp = 2(y - yp) * (-1)
# dl_dt1 = dl_dyp * sig_grad(t1) 
# dl_dw1 = dl_dt1 * x1
# dl_db1 = dl_dt1 * 1
# dl_dx1 = dl_dt1 * w1
# dl_dt0 = dl_dx1 * sig_grad(t0)


import numpy as np

class sigmoid:
    def __init__(self):

        self.out = None
        
    def forward(self, x):
        out =  1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, grad):
        self.grad = grad * (self.out) * (1 - self.out)
        return self.grad

    
class Linear:
    def __init__(self, in_feat, out_feat):
        self.W = np.random.randn(in_feat, out_feat)
        self.b = np.random.randn(out_feat)
        self.x = None
        self.grad_x = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        out = x @ self.W + self.b
        return out
    
    def backward(self, grad):
        self.grad_w = grad @ self.x.T
        self.grad_b = grad 
        self.grad_x = self.W.T @ grad 

        return self.grad_x
    

class NN:
    def __init__(self, in_feat, out_feat, hidden_dim):

        self.linear_1 = Linear(in_feat, hidden_dim)
        self.sigmoid_1 = self.sigmoid()

        self.linear_2 = Linear(hidden_dim, out_feat)
        self.sigmoid_2 = self.sigmoid_2


    def forward(self, x):

        out = self.sigmoid_1.forward(self.linear_1.forward(x))
        out = self.sigmoid_2.forward(self.linear_2.forward(out))

        return out

    def backward(self, grad):

        grad = self.sigmoid_2.backward(grad)
        grad = self.linear_2.backward(grad)
        grad = self.sigmoid_1(grad)
        grad = self.linear_1.backward(grad)

        



