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

def generateFeatureMatrix(adjMat, D):
  matrix = []
  for nodeIndex in range(0, len(adjMat)):
    processedNodes = []
    featureVector = getFeatureVectorOfNode(adjMat, D, nodeIndex, 0,  processedNodes)
    if featureVector == None:
      matrix.append([])
    else:
      matrix.append(featureVector)
  return matrix

#  for index1 in range(0, len(adjMat)):
#    featureMatrix = []
#    adjNodes = getAdjacentNode(adjMat, index1)
#    degreeNode = getDegreeNode(D, index1)
#    featureMatrix.append([degreeNode])
#    vectorPrevious = featureMatrix[len(featureMatrix) - 1][:]
#    for index2 in range(0, len(adjNodes)):
#      vectorPreviousToAppend = vectorPrevious[:]
#      nodeDegree = getDegreeNode(D, adjNodes[index2])
#      vectorPreviousToAppend.append(nodeDegree)
#      featureMatrix.append(vectorPreviousToAppend)
#    yield featureMatrix

def getFeatureVectorOfNode(adjMat, D, index, depth, processedNodes):
  if depth > 4: # or index in processedNodes:
    return None
  adjNodes = getAdjacentNode(adjMat, index)
  degreeNode = getDegreeNode(D, index)
  processedNodes.append(index)
  for indexAdj in range(0, len(adjNodes)):
    newNodeDegree = getFeatureVectorOfNode(adjMat, D, adjNodes[indexAdj], depth + 1, processedNodes)
    if newNodeDegree != None:
      return [degreeNode] + newNodeDegree
    else:
      return [degreeNode]

if __name__ == "__main__":
  #adjMat_dir, D_in, D_out, D, adjMat_undir = graphgenerate.generateRandomGraph(4, 4)
  adjMat_dir = [[1, 0, 0, 1,],
                [0, 0, 0, 0,], 
                [0, 0, 0, 0,],
                [0, 0, 0, 0,]]
  D_in = [[1, 0, 0, 0,],
          [0, 0, 0, 0,],
          [0, 0, 0, 0,],
          [0, 0, 0, 1,]]
  D_out = [[2, 0, 0, 0,], 
           [0, 0, 0, 0,], 
           [0, 0, 0, 0,], 
           [0, 0, 0, 0,]]
  D = [[3, 0, 0, 0,],
       [0, 0, 0, 0,],
       [0, 0, 0, 0,],
       [0, 0, 0, 1,]]
  adjMat_undir = [[1, 0, 0, 1,],
                  [0, 0, 0, 0,],
                  [0, 0, 0, 0,],
                  [1, 0, 0, 0,]]
  #print "adjMat_dir\n", adjMat_dir # adjancency matrix
  #print "D_in\n", D_in # diagonal degree matrix for incoming edges
  #print "D_out\n", D_out # diagonal degree matrix for outgoing edges
  #print "D\n", D # diagonal degree matrix D_in + D_out
  #print "adjMat_undir\n", adjMat_undir

  print generateFeatureMatrix(adjMat_undir, D)
