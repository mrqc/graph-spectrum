import graphgenerate
import numpy
import pprint

def degrees(adjMat_dir, adjMat_undir):
  D_in = numpy.zeros((len(adjMat_dir), len(adjMat_dir)))
  D_out = numpy.zeros((len(adjMat_dir), len(adjMat_dir)))
  D = numpy.zeros((len(adjMat_undir), len(adjMat_undir)))
  I = numpy.identity(len(adjMat_dir))
  for nodeIndex in range(0, len(adjMat_dir)):
    adjV1 = adjMat_dir[nodeIndex]
    adjV2 = numpy.transpose(adjMat_dir)[nodeIndex]
    D_out[nodeIndex][nodeIndex] = int(numpy.sum(adjV1))
    D_in[nodeIndex][nodeIndex] = int(numpy.sum(adjV2))
  for nodeIndex in range(0, len(adjMat_undir)):
    adjV = adjMat_undir[nodeIndex]
    D[nodeIndex][nodeIndex] = int(numpy.sum(adjV))
  return D_in, D_out, D

def getAdjacentNode(adjMat, nodeIndex):
  adjIndexes = []
  for index1 in range(0, len(adjMat)):
    if adjMat[nodeIndex][index1] == 1:
      adjIndexes.append(index1)
  return adjIndexes

def getDegreeNode(D, nodeIndex):
  return int(D[nodeIndex][nodeIndex])

def getUndirectedAdj(directedAdjMat):
  undirectedAdjMat = numpy.zeros((len(directedAdjMat), len(directedAdjMat[0])))
  for index1 in range(0, len(directedAdjMat)):
    for index2 in range(0, len(directedAdjMat[index1])):
      if directedAdjMat[index1][index2] == 1 or directedAdjMat[index2][index1] == 1:
        undirectedAdjMat[index1][index2] = undirectedAdjMat[index2][index1] = int(1)
  return undirectedAdjMat

def generateFeatureMatrix(adjMat, D):
  matrix = []
  for nodeIndex in range(0, len(adjMat)):
#    print "processing node " + str(nodeIndex)
    processedNodes = []
    featureMatrix = getFeatureMatrixOfNode(adjMat, D, nodeIndex, 0,  processedNodes)
    matrix.append(featureMatrix)
  return matrix

maxDepth = 2

def getFeatureMatrixOfNode(adjMat, D, index, depth, processedNodes):
  if depth > maxDepth:# or index in processedNodes:
#    print "not performing " + str(index) + " in " + str(depth)
    return None
#  print "features of node index: " + str(index) + " in depth " + str(depth)
  adjNodes = getAdjacentNode(adjMat, index)
#  print adjNodes
#  print adjNodes
  degreeNode = getDegreeNode(D, index)
#  print degreeNode
  processedNodes.append(index)
  matrix = [degreeNode]
  for indexAdj in range(0, len(adjNodes)):
    newNodeDegree = getFeatureMatrixOfNode(adjMat, D, adjNodes[indexAdj], depth + 1, processedNodes)
    if newNodeDegree != None:
      matrix.append(newNodeDegree)
  return matrix

def processMatrix(ele, vec, matrix):
  for ind in range(0, len(ele)):
    if not isinstance(ele[ind], list):
      #print vec + [ele[ind]]
      vec = vec + [ele[ind]]
      matrix.append(vec)
    else:
      processMatrix(ele[ind], vec, matrix)


