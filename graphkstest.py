import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.mlab import PCA
import graphgenerate
import graphfeature

def softmax(x):
  eX = np.exp(x - np.max(x))
  return eX / eX.sum()

"""
plt.figure()
for ind1 in range(0, len(nodesSoftmax)):
  plt.subplot(length, length, ind1 + 1)
  plt.ylim((0, 1))
  plt.xlim((0, 1))
  for ind2 in range(0, len(nodesSoftmax[ind1])):
    plt.plot(np.arange(0, len(nodesSoftmax[ind1][ind2])), nodesSoftmax[ind1][ind2], '-o', markersize=4)
plt.show()

print ""
print "if statistic is small or pvalue high, then F(x)=G(x)"
print ""
print "ks test for decreasing distri"
print stats.ks_2samp(softmaxY1, softmaxY3)
print ""
print "ks test for stable distri"
print stats.ks_2samp(softmaxY2, softmaxY4)
"""
