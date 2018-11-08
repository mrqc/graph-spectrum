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

adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(6, 6)
nodes = []
nodesSoftmax = []
maxNodes = 0
for matrix in graphfeature.generateFeatureMatrix(adjMat_undir, D):
  print matrix
  nodes.append([])
  nodesSoftmax.append([])
  for vector in matrix:
    vector = np.sort(vector)[::-1]
    nodesSoftmax[len(nodes) - 1].append(softmax(vector))
    nodes[len(nodes) - 1].append(vector)
    if len(nodes[len(nodes) - 1]) > maxNodes:
      maxNodes = len(nodes[len(nodes) - 1])
print nodesSoftmax
quit()


length = 0
if maxNodes > len(nodes):
  length = np.ceil(np.sqrt(maxNodes))
else:
  length = np.ceil(np.sqrt(len(nodes)))
plt.figure()
for ind1 in range(0, len(nodesSoftmax)):
  plt.subplot(length, length, ind1 + 1)
  plt.ylim((0, 1))
  plt.xlim((0, 1))
  for ind2 in range(0, len(nodesSoftmax[ind1])):
    plt.plot(np.arange(0, len(nodesSoftmax[ind1][ind2])), nodesSoftmax[ind1][ind2], '-o', markersize=4)
plt.show()

quit()
minLen1 = np.ndarray.min(np.array([len(node1[0]), len(node2[0])]))
minLen2 = np.ndarray.min(np.array([len(node1[1]), len(node2[1])]))

softmaxY1 = softmax(node1[0][0:minLen1])
softmaxY2 = softmax(node1[1][0:minLen2])
softmaxY3 = softmax(node2[0][0:minLen1])
softmaxY4 = softmax(node2[1][0:minLen2])

X1 = np.arange(0, minLen1)
X2 = np.arange(0, minLen2)
X3 = np.arange(0, minLen1)
X4 = np.arange(0, minLen2)

#plt.style.use("seaborn-whitegrid")
plt.figure()

plt.subplot(4, 4, 1)
plt.plot(X1, softmaxY1, '-o', markersize=4)
plt.plot(X2, softmaxY2, '-o', markersize=4)

plt.subplot(4, 4, 2)
plt.plot(X3, softmaxY3, '-o', markersize=4)
plt.plot(X4, softmaxY4, '-o', markersize=4)

print ""
print ""
print "if statistic is small or pvalue high, then F(x)=G(x)"
print ""
print "ks test for decreasing distri"
print stats.ks_2samp(softmaxY1, softmaxY3)
print ""
print "ks test for stable distri"
print stats.ks_2samp(softmaxY2, softmaxY4)

