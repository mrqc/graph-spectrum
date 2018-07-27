import graphgenerate
import numpy

def degrees(adjMat):
  D_in = numpy.zeros((len(adjMat), len(adjMat)))
  D_out = numpy.zeros((len(adjMat), len(adjMat)))
  I = numpy.identity(len(adjMat))
  for nodeIndex in range(0, len(adjMat)):
    adjV1 = adjMat[nodeIndex]
    adjV2 = numpy.transpose(adjMat)[nodeIndex]
    D_out[nodeIndex][nodeIndex] = int(numpy.sum(adjV1))
    D_in[nodeIndex][nodeIndex] = int(numpy.sum(adjV2))
  D = numpy.add(D_in, D_out)
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

adjMat_dir = numpy.fromfunction(graphgenerate.adjFun, (4, 4))
(D_in, D_out, D) = degrees(adjMat_dir)
adjMat_undir = getUndirectedAdj(adjMat_dir)
print "adjMat_dir\n", adjMat_dir # adjancency matrix
print "D_in\n", D_in # diagonal degree matrix for incoming edges
print "D_out\n", D_out # diagonal degree matrix for outgoing edges
print "D\n", D # diagonal degree matrix D_in + D_out
print "adjMat_undir\n", adjMat_undir

def generateFeatureMatrix(featureMatrix, ):
  

for index1 in range(0, len(adjMat_undir)):
  featureMatrix = []
  print "\nnode index", index1
  adjNodes = getAdjacentNode(adjMat_undir, index1)
  print "adj nodes indexes", adjNodes
  degreeNode = getDegreeNode(D, index1)
  print "degree of node", degreeNode
  featureMatrix.append([degreeNode])
  vectorPrevious = featureMatrix[len(featureMatrix) - 1][:]
  for index2 in range(0, len(adjNodes)):
    vectorPreviousToAppend = vectorPrevious[:]
    vectorPreviousToAppend.append(getDegreeNode(D, index2))
    featureMatrix.append(vectorPreviousToAppend)
  print featureMatrix
  break
