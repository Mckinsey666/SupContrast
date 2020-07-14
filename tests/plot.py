import numpy as np 
import matplotlib.pyplot as plt

p = np.load('../p_comp.npy')

if_l = np.load('../info_loss.npy')
c_l = np.load('../cls_loss.npy')

def sigmoid(p, t = 1):
    return 1 / (1 + np.exp(-p / t))

fig = plt.figure()

aug = ['crop', 'flip', 'jitter', 'gray']
for i in range(len(p)):
    plt.plot([0.5]+list(p[i]), label=aug[i])
plt.legend()
#plt.plot(c_l)
#plt.show()
plt.show()