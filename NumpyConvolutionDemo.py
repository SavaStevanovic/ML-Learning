import numpy as np
import scipy.signal


def conv1d(x, w, p=0, s=1):
    p = max(0, p)
    w_rot = np.array(w[::-1])
    padding = np.zeros(shape=p)
    x_padded = np.concatenate([padding, x, padding])

    res = []
    for i in range(0, int((len(x_padded)-len(w_rot))/s)+1, s):
        res.append(np.dot(x_padded[i:i+len(w_rot)], w_rot))

    return np.array(res)


def conv2d(X, W, p=(0, 0), s=(1, 1)):
    W_rot = np.array(W)[::-1,::-1]
    X_orig = np.array(X)
    X_padded = np.zeros(shape=(2*p[0]+X_orig.shape[0], 2*p[1]+X_orig.shape[1]))
    X_padded[p[0]:p[0]+X_orig.shape[0], p[1]:p[1]+X_orig.shape[1]] = X_orig

    res = []

    for i in range(0, int((X_padded.shape[0]-W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1]-W_rot.shape[1])/s[1])+1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                             j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))
    return(np.array(res))


x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

print('Conv1d Implementation:',  conv1d(x, w, p=2, s=1))
print('Numpy Results:', np.convolve(x, w, mode='same'))

X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

print('Conv2d Implementation:\n', conv2d(X, W, p=(1, 1), s=(1, 1)))
print('SciPy Results:\n', scipy.signal.convolve2d(X, W, mode='same'))
