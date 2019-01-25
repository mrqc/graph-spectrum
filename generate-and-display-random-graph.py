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
import warnings
import sys
warnings.simplefilter("ignore")
nodeCount = 10

def sqr(v1, v2):
  error = 0
  if len(v1) != len(v2):
    raise Exception("Length not equal")
  for index in range(0, len(v1)):
    error = error + numpy.power(numpy.absolute(v1[index] - v2[index]), graphfeature.maxDepth - index + 1)
  return error # / float(len(v1))

def ppArray(a, depth=0):
  for ele in a:
    if isinstance(ele, list):
      if isinstance(ele[0], list):
        print ' ' * (depth * 3), "["
        ppArray(ele, depth + 1)
        print ' ' * (depth * 3), "]"
      else:
        print ' ' * (depth * 3), str(ele)
    else:
      print ' ' * (depth * 3), str(ele) + ","

def comparePart(node1Features, node2Features):
  indexesUsed = []
  bestIndexes = []
  for index1 in range(0, len(node1Features)):
    dists = []
    for index2 in range(0, len(node2Features)):
      #print node1Features[index1]
      #print node2Features[index2]
      v1 = numpy.log(node1Features[index1])
      v2 = numpy.log(node2Features[index2])
      for index in range(0, len(v1)):
        if v1[index] == numpy.NINF:
          v1[index] = 0
      for index in range(0, len(v2)):
        if v2[index] == numpy.NINF:
          v2[index] = 0
      test = sqr(v1, v2)
      dist = test
      dists.append(dist)
    maxIndex = -1
    for index in range(0, len(dists)):
      if maxIndex == -1 and index not in indexesUsed:
        maxIndex = index
      elif dists[maxIndex] > dists[index] and index not in indexesUsed:
        maxIndex = index
    if maxIndex != -1:
      indexesUsed.append(maxIndex)
      bestIndexes.append((index1, maxIndex, dists[maxIndex]))
  return bestIndexes

def compare(node1Features, node2Features):
  if node1Features[0][0] == 0:
    if node2Features[0][0] == 0:
      return ([(0.0, 0.0)], 0.0)
    else:
      return ([(0.0, 0.0)], numpy.inf)
  elif node2Features[0][0] == 0:
    return ([(0.0, 0.0)], numpy.inf)
  length = len(node1Features[0]) if len(node1Features[0]) < len(node2Features[0]) else len(node2Features[0])
  node1FeaturesLocal = node1Features
  node2FeaturesLocal = node2Features
  for index in range(0, len(node1FeaturesLocal)):
    node1FeaturesLocal[index] = node1FeaturesLocal[index][0:length]
  for index in range(0, len(node2FeaturesLocal)):
    node2FeaturesLocal[index] = node2FeaturesLocal[index][0:length]
  indexes1 = comparePart(node1FeaturesLocal, node2FeaturesLocal)
  indexes2 = comparePart(node2FeaturesLocal, node1FeaturesLocal)
  diff1 = diff(indexes1)
  diff2 = diff(indexes2)
  return (indexes1, diff1) if diff1 < diff2 else (indexes2, diff2)

def diff(v):
  diffVal = 0
  for data in v:
    diffVal = diffVal + numpy.power(data[2], 2)
  diffVal = diffVal / len(v)
  return diffVal

def euclideanDist(v1, v2):
  return distance.euclidean(v1, v2)

def printPairs(pairs):
  for pair in pairs:
    if pair[0] != pair[1]:
      print pair[0], "vs.", pair[1], "=", pair[2]

def calcDiffs(adjMat, DegMat):
  featureMatrix = graphfeature.generateFeatureMatrix(adjMat, DegMat)
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
        finishedMatrix[len(finishedMatrix) - 1].append(vec)
      elif len(vec) == 1:
        if vec[0] == 0:
          finishedMatrix[len(finishedMatrix) - 1].append([0])
  normalizedMatrix = []
  for matrix in finishedMatrix:
    normalizedMatrix.append([])
    for vector in matrix:
      normalizedMatrix[len(normalizedMatrix) - 1].append(vector)

  pairs = []
  keyPairs = {}
  for index1 in range(0, len(normalizedMatrix)):
    for index2 in range(index1 + 1, len(normalizedMatrix)):
      cVal = compare(normalizedMatrix[index1], normalizedMatrix[index2])
      pairs.append((index1, index2, cVal[1]))
      keyPairs[(index1, index2)] = cVal[1]

  pairs.sort(key=lambda x: x[2], reverse=False)
  return pairs, keyPairs

if __name__ == "__main__":
  adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(nodeCount, nodeCount)
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

#  adjMat_undir = numpy.array([
#    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
#  D = numpy.array([
#    [12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

#  D = numpy.array([
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0]])
#  adjMat_undir = numpy.array([
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0]])
  plt.subplot(221)
  graphdisplay.renderDiGraphFromAdj(adjMat_dir)
  plt.subplot(222)
  graphdisplay.renderGraphFromAdj(adjMat_undir)
  #undirected graph
  pairs1, keyPairs1 = calcDiffs(adjMat_undir, D)
  print "Undirected Graph Differences between nodes"
  printPairs(pairs1)

  #directed graph
  pairs2, keyPairs2 = calcDiffs(adjMat_dir, D_out)
  pairs3, keyPairs3 = calcDiffs(adjMat_dir, D_in)
  pairs4 = []
  #printPairs(pairs2)
  #print "---"
  #printPairs(pairs3)
  #for ele2 in pairs2:
  #  cVal1 = ele2[2]
  #  cVal2 = keyPairs3[(ele2[0], ele2[1])]
  #  if cVal1 != numpy.Inf and cVal2 != numpy.Inf:
  #    pairs4.append((ele2[0], ele2[1], cVal1, cVal2))
  #print "Directed Graph Differences between nodes"
  #for index1 in range(0, len(pairs4)):
  #  for index2 in range(0, len(pairs4)):
  #    x1 = pairs4[index1][2]
  #    y1 = pairs4[index1][3]
  #    x2 = pairs4[index2][2]
  #    y2 = pairs4[index2][3]
  #    print pairs4[index1][euclideanDist([x1, y1], [x2, y2])
  #plt.show()

