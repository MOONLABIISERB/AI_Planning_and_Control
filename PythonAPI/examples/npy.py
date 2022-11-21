import numpy as np
x = np.load('a.npy',allow_pickle=True)
import matplotlib.pyplot as plt

x_mean = x[:, 0]
x_ =[]
for i in x_mean:
    x_.append(np.linalg.norm(i[:-1] - i[1:], axis=1))
x_ = np.array(x_)
fig = plt.figure()
plt.plot(np.arange(0, len(x_)), x_.mean(axis=1))
plt.plot(np.arange(0,len(x_)), x[:, 1])
plt.ylim(2.5, 2.7)

fig2 = plt.figure()
plt.plot(np.arange(0, len(x_)), x[:, 2])
plt.plot(np.arange(0, len(x_)), 3.6*x[:, 3])
plt.plot(np.arange(0, len(x_)), 3.6*x[:, 4])
plt.legend(['acc', 't_s', 's'])
plt.show()