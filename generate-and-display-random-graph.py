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
  print "comparing -------------------------"
  print node1Features, node2Features
  print "-----------------------------------"
  plt.subplot(223)
  plt.xlabel("statistic (should be low)")
  plt.ylabel("pvalue (should be high)")
  axes = plt.gca()
  axes.set_xlim([0, 1.1])
  axes.set_ylim([0, 1.1])
  xValues = []
  yValues = []
  font = { "family": "sans-serif", "size": 10 }
  indexesUsed = []
  bestIndexes = []
  for index1 in range(0, len(node1Features)):
    dists = []
    distsIndexes = []
    print "-" * 20
    for index2 in range(0, len(node2Features)):
      test = stats.ks_2samp(node1Features[index1], node2Features[index2])
      print "compareing vector " + str(index1) + " with " + str(index2)
      print node1Features[index1], node2Features[index2], test
      plt.plot([test.statistic], [test.pvalue], color="red", marker="o", markersize=8 - 1 * index1, alpha=0.5)
      normVector = (0, 1)
      dist = euclideanDist(normVector, (test.statistic, test.pvalue))
      dists.append(dist)
      distsIndexes.append((index1, index2))
      #plt.text(test.statistic, test.pvalue + index1 * 0.08 + index2 * 0.08, str(index1) + "-" + str(index2), fontdict=font)
    maxIndex = -1
    for index in range(0, len(dists)):
      if maxIndex == -1 and index not in indexesUsed:
        maxIndex = index
      elif dists[maxIndex] > dists[index] and index not in indexesUsed:
        maxIndex = index
    # maxIndex now best choice
    if maxIndex != -1:
      indexesUsed.append(maxIndex)
      print "best is " + str(index1) + " " + str(index2)
      bestIndexes.append((index1, index2))
  return bestIndexes

def euclideanDist(v1, v2):
  return distance.euclidean(v1, v2)

if __name__ == "__main__":

  adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(6, 6)
#  adjMat_undir = numpy.array([
#   [0, 1, 0, 1, 1, 0],
#   [1, 0, 1, 0, 0, 1],
#   [0, 1, 0, 1, 0, 0],
#   [1, 0, 1, 0, 0, 1],
#   [1, 0, 0, 0, 0, 0],
#   [0, 1, 0, 1, 0, 0]])
#  D = numpy.array([
#   [3, 0, 0, 0, 0, 0],
#   [0, 3, 0, 0, 0, 0],
#   [0, 0, 2, 0, 0, 0],
#   [0, 0, 0, 3, 0, 0],
#   [0, 0, 0, 0, 1, 0],
#   [0, 0, 0, 0, 0, 2]])

  adjMat_undir = numpy.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
  
  D = numpy.array([
    [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

#  print "adjMat_dir"
#  print adjMat_dir
  print "adjMat_undir"
  print adjMat_undir
#  print "D"
#  print D
#  print "D_out"
#  print D_out
#  print "D_in"
#  print D_in

  plt.subplot(221)
  graphdisplay.renderGraphFromAdj(adjMat_undir)
  #plt.subplot(222)
  #graphdisplay.renderDiGraphFromAdj(adjMat_dir)

  featureMatrix = graphfeature.generateFeatureMatrix(adjMat_undir, D)
  print featureMatrix
#  print "featureMatrix"
#  print featureMatrix

  finishedMatrix = []
  for featureIndex in range(0, len(featureMatrix)):
    finishedMatrix.append([])
    matrix = []
    graphfeature.processMatrix(featureMatrix[featureIndex], [], matrix)
    length = 0
    for vec in matrix:
      if len(vec) > length:
        length = len(vec)

    for vec in matrix:
      if len(vec) == length:
        #finishedMatrix[len(finishedMatrix) - 1].append(sorted(vec, reverse = True))
        finishedMatrix[len(finishedMatrix) - 1].append(vec)
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

#  print "softmax matrix"
#  print softmaxMatrix
#  quit()

  #for index1 in range(0, len(softmaxMatrix)):
  #  for index2 in range(index1 + 1, len(softmaxMatrix)):
  #    compare(softmaxMatrix[index1], softmaxMatrix[index2])
  indexes = compare(softmaxMatrix[0], softmaxMatrix[13])
  print indexes


  plt.show()

