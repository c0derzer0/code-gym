# Attempt 1 — 25 May, 2026
# Started: 3:40pm
# finished (first pass runs): 4:30pm (~50 min)
# finished after correction hints: 4:55pm (~75 min total)
# First pass (runs without error): NO — missing lr, inverted bias correction
# Correct (passes tests): yes — matches torch.optim.Adam over 50 steps to 1e-5 on both params
# Stuck on: bias correction mechanics (where to apply it, which direction)
# Notes (bugs caught + fixes, with hints):
#   - missing the parameter update entirely on first pass (computed m,v but never touched p)
#   - bias correction was inverted (multiplying by (1 - β^t) instead of dividing) AND stored
#     back into self.m / self.v, corrupting the EMA state. fix: keep self.m, self.v as raw EMA;
#     compute m_hat = m / (1 - β₁^t), v_hat = v / (1 - β₂^t) as temps used only for the update.
#   - missing self.lr in the parameter update — Adam was taking effective unit steps,
#     params drifted to ~10 instead of staying near init. textbook giveaway.
#   - Python scoping bug in test loop: inner `for model, opt in [[mlp, opt], ...]` reassigned
#     `opt` to opt_ref after first inner iter, so own Adam ran exactly once across 50 outer
#     iters. fix: rename inner var to `opti` so outer `opt` isn't shadowed.
#   - MSELoss arg order was (y, out); convention is (input, target) → (out, y). symmetric for
#     MSE but bad habit for asymmetric losses.
# Followups for attempt 2:
#   - Zero-init self.m, self.v in __init__ via dict comprehension over self.params; drops the
#     if-in/else branch in step().
#   - Drop the lazy-vs-eager init choice in favor of eager — cleaner step() loop.
import torch
import torch.nn as nn
import math 

class Adam():
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):     
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.params = list(params)
        self.m = {}
        self.v = {}
        self.t = 1

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    m_bias = 1 / (1-math.pow(self.beta_1, self.t))
                    v_bias = 1 / (1-math.pow(self.beta_2, self.t))
                    if p in self.m:
                        self.m[p] = (self.beta_1 * self.m[p] + (1-self.beta_1) * p.grad) 
                        self.v[p] = (self.beta_2 * self.v[p] + (1-self.beta_2) * (p.grad)*(p.grad)) 
                    else:
                        self.m[p] = ((1-self.beta_1) * p.grad) 
                        self.v[p] = ((1-self.beta_2) * p.grad * p.grad) 

                    p.data = p.data - self.lr * self.m[p] * m_bias  / (torch.sqrt(self.v[p] * v_bias) + self.eps)

        self.t += 1
        # print(self.m)
        # print(self.v)


    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class MLP(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.fc1 = nn.Linear(in_feat, out_feat)

    def forward(self, x):
        return self.fc1(x)

  
x = torch.randn((4))
y = torch.tensor([1, 1])

torch.manual_seed(0)
mlp = MLP(4, 2)
opt = Adam(mlp.parameters())

torch.manual_seed(0)
mlp_ref = MLP(4, 2)
opt_ref = torch.optim.Adam(mlp_ref.parameters())


for step in range(50):

    for model, opti in [[mlp, opt], [mlp_ref, opt_ref]]:
        out = model(x)
        opti.zero_grad()
        loss =  nn.MSELoss()(out, y)
        loss.backward()
        opti.step()

for p, p_ref in zip(mlp.parameters(), mlp_ref.parameters()):
    print(torch.allclose(p, p_ref, atol=1e-5))