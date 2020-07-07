import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim = -1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

if __name__ == '__main__':
    # input is log of normalized probability
    t = torch.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 10)
    v = Variable(t)
    print(v)
    c = gumbel_softmax(v, 0.8)
    print(c)