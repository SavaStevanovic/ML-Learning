import matplotlib.pyplot as plt
import numpy as np


def gini(p):
    return 1-p**2-(1-p)**2


def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


def error(p):
    return 1-np.max([p, 1-p])


x = np.arange(0, 1, 0.01)
entrpy = [entropy(p) for p in x]
entrpy_scaled = [p/2 for p in entrpy]
gini = [gini(p) for p in x]
error = [error(p) for p in x]

fig = plt.figure()
ax = plt.subplot(111)
for d, lab, ls, c in zip([entrpy_scaled, entrpy, gini, error],
                              ['Entropy' (scaled), 'Entropy',
                               'Gini Impurity',
                               'Misclassification Error'],
                              ['-', '-', '--', '-.'],
                              ['black', 'lightgray', 'red', 'green', 'cyan']):
                              line = ax.plot(x, d, label=lab,
                                             color=c, linestyle=ls, lw=2)
ax.legend(loc='upper center', bbox_to_anchor=(
    0.5, 1.15), ncol=5, fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1]) 
plt.xlabel('p(i=1)') 
plt.ylabel('Impurity Index') 
plt.show()
