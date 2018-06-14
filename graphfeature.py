import graphgenerate
import numpy

def degrees(adjMat):
  D_in = numpy.zeros((len(adjMat), len(adjMat)))
  D_out = numpy.zeros((len(adjMat), len(adjMat)))
  for nodeIndex in range(0, len(adjMat)):
    adjV1 = adjMat[nodeIndex]
    adjV2 = numpy.transpose(adjMat)[nodeIndex]
    D_in[nodeIndex][nodeIndex] = numpy.sum(adjV1)
    D_out[nodeIndex][nodeIndex] = numpy.sum(adjV2)
  D = numpy.add(D_in, D_out)
  return D_in, D_out, D

adjMat = numpy.fromfunction(graphgenerate.adjFun, (50, 50))
(D_in, D_out, D) = degrees(adjMat)
print D_in
print D_out
print D
