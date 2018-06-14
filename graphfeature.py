import graphgenerate
import numpy

def degrees(adjMat):
  D_in = numpy.zeros((len(adjMat), len(adjMat)))
  D_out = numpy.zeros((len(adjMat), len(adjMat)))
  I = numpy.identity(len(adjMat))
  for nodeIndex in range(0, len(adjMat)):
    adjV1 = adjMat[nodeIndex]
    adjV2 = numpy.transpose(adjMat)[nodeIndex]
    D_in[nodeIndex][nodeIndex] = numpy.sum(adjV1)
    D_out[nodeIndex][nodeIndex] = numpy.sum(adjV2)
  D = numpy.add(D_in, D_out)
  selfLoopMatrix = numpy.multiply(adjMat, I)
  D_complete = numpy.subtract(D, selfLoopMatrix)
  return D_in, D_out, D, D_complete

adjMat = numpy.fromfunction(graphgenerate.adjFun, (50, 50))
(D_in, D_out, D, D_complete) = degrees(adjMat)
print adjMat # adjancency matrix
print D_in # diagonal degree matrix for incoming edges
print D_out # diagonal degree matrix for outgoing edges
print D # diagonal degree matrix D_in + D_out
print D_complete # diagonal degree matrix excluding self loops
