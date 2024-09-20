'''
Generate wave data.
'''

import numpy as np
import scipy.io

L = 62832
N = 10000
X = np.zeros((N, L))
Y = np.zeros((N, 4))
dt = 0.0001

f1 = 1
f2 = 2**4
f3 = 2**8
f4 = 2**12
s1 = lambda x: np.cos(f1 * x)
s2 = lambda x: np.cos(f2 * x)
s3 = lambda x: np.cos(f3 * x)
s4 = lambda x: np.cos(f4 * x)
w1 = s1(dt * np.arange(L))
w2 = s2(dt * np.arange(L))
w3 = s3(dt * np.arange(L))
w4 = s4(dt * np.arange(L))

grid = np.arange(0.1, 1.1, 0.1)

counter = 0
for i in grid:
    for j in grid:
        for k in grid:
            for l in grid:
                X[counter, :] += i * w1
                X[counter, :] += j * w2
                X[counter, :] += k * w3
                X[counter, :] += l * w4
                Y[counter, 0] = i
                Y[counter, 1] = j
                Y[counter, 2] = k
                Y[counter, 3] = l
                counter += 1

N_test = 10000
X_test = np.zeros((N_test, L))
Y_test = np.zeros((N_test, 4))

for i in range(N_test):
    a1 = np.random.rand()
    a2 = np.random.rand()
    a3 = np.random.rand()
    a4 = np.random.rand()
    X_test[i, :] += a1 * w1
    X_test[i, :] += a2 * w2
    X_test[i, :] += a3 * w3
    X_test[i, :] += a4 * w4
    Y_test[i, 0] = a1
    Y_test[i, 1] = a2
    Y_test[i, 2] = a3
    Y_test[i, 3] = a4

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
Y = Y.astype(np.float32)
Y_test = Y_test.astype(np.float32)

scipy.io.savemat('waves.mat', {'X': X, 'Y': Y, 'X_test': X_test, 'Y_test': Y_test})