import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class AugSample(nn.Module):
    def __init__(self, n):
        super(AugSample, self).__init__()
        self.n = n
        #self.prob = Variable(torch.zeros(self.n), requires_grad = True)
        self.prob = Variable(torch.empty(self.n).fill_(0.5), requires_grad = True)
        self.temperature = 1

    def relaxed_bernoulli(self, p, temperature = 1, eps = 1e-20):
        """
        Args:
            p: [N] tensor indicating bernoulli probability
        Returns:
            c: [N] with 0, 1 values indicating bernoulli sample (hard)
        """
        p = torch.clamp(p, 0, 1)
        #p = torch.sigmoid(p / 10) # raw to (0, 1) prob
        u = torch.empty(p.shape).uniform_(0, 1)
        q = torch.log(p/(1 - p + eps) + eps) + torch.log(u / (1 - u + eps) + eps)
        y_soft = torch.sigmoid(q / temperature)
        y_hard = torch.where(y_soft > 0.5, torch.ones(y_soft.shape), torch.zeros(y_soft.shape))
        return y_hard - y_soft.detach() + y_soft

    def forward(self):
        return self.relaxed_bernoulli(self.prob, temperature = self.temperature)



if __name__ == '__main__':
    pm = AugSample(5).cuda()
    crit = nn.MSELoss().cuda()

    optimzer = optim.SGD([pm.prob], lr=1)
    target = torch.randn(2,2).cuda()


    for i in range(10000):
        s = pm().cuda()
        x = target.clone()
        for i in range(4):
            x = x * 1-s[i]
        x = x * s[-1]

        loss = crit(x, target)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
    print(torch.sigmoid(pm.prob))