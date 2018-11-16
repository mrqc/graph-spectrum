import numpy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import motifs
import graphgenerate
import graphdisplay
import graphfeature
import graphkstest
from scipy.spatial import distance
from scipy import stats
import matplotlib.cm as cm

def compare(node1Features, node2Features):
  plt.subplot(223)
  plt.xlabel("statistic (should be low)")
  plt.ylabel("pvalue (should be high)")
  xValues = []
  yValues = []
  colors =  ["black", "red", "gold", "aliceblue", "royalblue", "navy", "blue", "grey", "lightcoral", "orange", "olive", "c", "skyblue", "magenta", "indianread", "darkred", "pink"]
  for index1 in range(0, len(node1Features)):
    for index2 in range(0, len(node2Features)):
      test = stats.ks_2samp(node1Features[index1], node2Features[index2])
      print "compare"
      print node1Features[index1], node2Features[index2]
      print test
      plt.plot([test.statistic], [test.pvalue], color=colors[index1 % len(colors)], marker="o", markersize=8 - 0.5 * index1, alpha=0.5)

def euclideanDist(v1, v2):
  return distance.euclidean(v1, v2)

adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(6, 6)

plt.subplot(221)
graphdisplay.renderDiGraphFromAdj(adjMat_dir)
plt.subplot(222)
graphdisplay.renderGraphFromAdj(adjMat_undir)

print "adjMat_dir"
print adjMat_dir
print "adjMat_undir"
print adjMat_undir
print "D"
print D
print "D_out"
print D_out
print "D_in"
print D_in

featureMatrix = graphfeature.generateFeatureMatrix(adjMat_undir, D)
print "featureMatrix"
print featureMatrix

finishedMatrix = []
for featureIndex in range(0, len(featureMatrix)):
  finishedMatrix.append([])
  matrix = []
  graphfeature.processMatrix(featureMatrix[featureIndex], [], matrix)
  for vec in matrix:
    if len(vec) == graphfeature.maxDepth + 1:
      finishedMatrix[len(finishedMatrix) - 1].append(sorted(vec, reverse = True))
    elif len(vec) == 1:
      if vec[0] == 0:
        finishedMatrix[len(finishedMatrix) - 1].append([0])

print "vector matrix"
print finishedMatrix


softmaxMatrix = []
for matrix in finishedMatrix:
  softmaxMatrix.append([])
  for vector in matrix:
    softmaxMatrix[len(softmaxMatrix) - 1].append(graphkstest.softmax(vector))

print "softmax matrix"
print softmaxMatrix

compare(softmaxMatrix[0], softmaxMatrix[2]) 

plt.show()

