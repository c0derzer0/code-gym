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
        self.grad = None
        
    def forward(self, x):
        out =  1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, grad):
        self.grad = grad * (self.out) * (1 - self.out)
        return self.grad

def bce_loss(y, pred):
    return - np.mean(y * np.log(pred) + (1-y) * np.log(1-pred))

def bce_grad(y, pred):
    return (pred - y) / y.shape[0]


    
class Linear:
    def __init__(self, in_feat, out_feat):
        self.W = np.random.randn(in_feat, out_feat)
        self.b = np.zeros(out_feat)
        self.x = None
        self.grad_x = None
        self.grad_w = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        out = x @ self.W + self.b
        return out
    
    def backward(self, grad):
        self.grad_w = self.x.T  @ grad  # grad - (N, out_feat), x - (N, in_feat)
        self.grad_b = grad.sum(axis=0)  # grad - (N, out_feat), b - ()
        self.grad_x = grad @ self.W.T  # grad - (N, out_feat), W - (in_feat, out_feat)

        return self.grad_x
    
    def step(self, lr):
        self.W -= lr * self.grad_w
        self.b -= lr * self.grad_b
    

class NN:
    def __init__(self, in_feat, out_feat, hidden_dim):

        self.linear_1 = Linear(in_feat, hidden_dim)
        self.sigmoid_1 = sigmoid()

        self.linear_2 = Linear(hidden_dim, out_feat)
        self.sigmoid_2 = sigmoid()

        self.grad = None

    def forward(self, x):

        out = self.sigmoid_1.forward(self.linear_1.forward(x))
        out = self.sigmoid_2.forward(self.linear_2.forward(out))

        return out

    def backward(self, grad):

        grad = self.linear_2.backward(grad)
        grad = self.sigmoid_1.backward(grad)
        grad = self.linear_1.backward(grad)

        return grad
    
    def step(self, lr):
        self.linear_2.step(lr)
        self.linear_1.step(lr)
    
    

def train():      

    N, in_feat, out_feat, hidden_dim = 50, 4, 1, 2
    X = np.random.randn(N, in_feat)
    y = (X.sum(axis=1, keepdims=True) > 0).astype(float)

    nn = NN(in_feat, out_feat, hidden_dim)
    

    mini_batch_size = 5
    n_batches = N // mini_batch_size
    print(n_batches)
    epochs = 100

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        for i in range(0, N, mini_batch_size):
            #print(X[i:i+mini_batch_size, :].shape)
            pred = nn.forward(X[i:i+mini_batch_size, :])
            #print(pred.shape)

            loss = bce_loss(y[i:i+mini_batch_size], pred)
            out_grad = bce_grad(y[i:i+mini_batch_size], pred)
            nn.backward(out_grad)
            nn.step(0.1)

        print(f'loss at {epoch} is {loss}')

train()



        
    


        



