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

def sqr(v1, v2):
  error = 0
  if len(v1) != len(v2):
    raise Exception("Length not equal")
  for index in range(0, len(v1)):
    error = error + numpy.power(numpy.absolute(v1[index] - v2[index]), graphfeature.maxDepth - index + 1)
  return error # / float(len(v1))

def compare(node1Features, node2Features):
  #print "comparing -------------------------"
  #print node1Features, node2Features
  #print "-----------------------------------"
  #plt.subplot(223)
  #plt.xlabel("statistic (should be low)")
  #plt.ylabel("pvalue (should be high)")
  #axes = plt.gca()
  #axes.set_xlim([0, 1])
  #axes.set_ylim([0, 1])
  #xValues = []
  #yValues = []
  #font = { "family": "sans-serif", "size": 10 }
  if node1Features[0][0] == 0:
    if node2Features[0][0] == 0:
      return [(0, 0, 0)]
    else:
      return [(0, 0, numpy.inf)]
  elif node2Features[0][0] == 0:
    return [(0, 0, numpy.inf)]

  indexesUsed = []
  bestIndexes = []
  for index1 in range(0, len(node1Features)):
    dists = []
    #print "-" * 20
    for index2 in range(0, len(node2Features)):
      v1 = numpy.log(node1Features[index1])
      v2 = numpy.log(node2Features[index2])
      #print "comparing vector " + str(index1) + " with " + str(index2)
      #test = stats.stats.ttest_ind(v1, v2)
      #plt.plot([test.statistic], [test.pvalue], color="red", marker="o", markersize=8 - 1 * index1, alpha=0.5)
      test = sqr(v1, v2)
      print v1, v2
      #print node1Features[index1]
      #print node2Features[index2]
      #print ['{0:.5f}'.format(_) for _ in v1]
      #print ['{0:.5f}'.format(_) for _ in v2]
      #print test
      #print node1Features[index1], node2Features[index2], test
      #plt.plot(index1, test, color="red", marker="o", markersize=8 - 1 * index1, alpha=0.5)

      #normVector = (0, 1)
      #dist = euclideanDist(normVector, (test.statistic, test.pvalue))
      dist = test
      dists.append(dist)
      #plt.text(test.statistic, test.pvalue + index1 * 0.08 + index2 * 0.08, str(index1) + "-" + str(index2), fontdict=font)
      #plt.text(index1, test, str(index1) + "-" + str(index2), fontdict=font)
    maxIndex = -1
#    if check:
#      print dists
    print dists
    for index in range(0, len(dists)):
      if maxIndex == -1 and index not in indexesUsed:
        maxIndex = index
      elif dists[maxIndex] > dists[index] and index not in indexesUsed:
        maxIndex = index
#    if check:
#      print maxIndex
    # maxIndex now best choice
    if maxIndex != -1:
      indexesUsed.append(maxIndex)
      #print "best is " + str(index1) + " " + str(maxIndex)
      bestIndexes.append((index1, maxIndex, dists[maxIndex]))
  return bestIndexes

def euclideanDist(v1, v2):
  return distance.euclidean(v1, v2)

if __name__ == "__main__":
  adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(7, 7)
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
#  adjMat_undir = numpy.array([
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0],
#    [0, 0, 0, 0]])

#  
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

#
#  print "adjMat_dir"
#  print adjMat_dir
#  print "adjMat_undir"
#  print adjMat_undir
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
  #print featureMatrix
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

  #print "vector matrix"
  #print finishedMatrix


  softmaxMatrix = []
  for matrix in finishedMatrix:
    softmaxMatrix.append([])
    for vector in matrix:
      softmaxMatrix[len(softmaxMatrix) - 1].append(vector)

#  print "softmax matrix"
#  print softmaxMatrix
#  quit()

  #for index1 in range(0, len(softmaxMatrix)):
  #  for index2 in range(index1 + 1, len(softmaxMatrix)):
  #    compare(softmaxMatrix[index1], softmaxMatrix[index2])
  def diff(v):
    diffVal = 0
    for data in v:
      diffVal = diffVal + numpy.power(data[2], 2)
    diffVal = diffVal / len(v)
    return diffVal

#  for index1 in range(0, len(softmaxMatrix)):
#    for index2 in range(0, len(softmaxMatrix)):
#      indexes = compare(softmaxMatrix[index1], softmaxMatrix[index2])
#      difference = diff(indexes)
#      if index1 == index2 and difference != 0:
#        print "-"
#        print softmaxMatrix[index1]
#        print softmaxMatrix[index2]
#        print "-"
#      print str(index1) + " vs " + str(index2) + ": " + str(difference)
  indexes01 = compare(softmaxMatrix[0], softmaxMatrix[1])
  print "-" * 20
  indexes10 = compare(softmaxMatrix[1], softmaxMatrix[0])
  print indexes01
  print indexes10

  #indexes1 = compare(softmaxMatrix[0], softmaxMatrix[13])
  #indexes2 = compare(softmaxMatrix[0], softmaxMatrix[14])
  #indexes3 = compare(softmaxMatrix[1], softmaxMatrix[14])
  #indexes4 = compare(softmaxMatrix[1], softmaxMatrix[13])
  #print indexes1
  #print indexes2
  #print indexes3
  #print indexes4

  #print "0 vs 13: " + str(diff(indexes1))
  #print "0 vs 14: " + str(diff(indexes2))
  #print "1 vs 14: " + str(diff(indexes3))
  #print "1 vs 13: " + str(diff(indexes4))
  plt.show()

