import graphgenerate
import numpy

def degrees(adjMat, nodeIndex):
  D_in = adjMat[nodeIndex]
  D_out = numpy.transpose(adjMat)[nodeIndex]
  return D_in, D_out, numpy.sum(D_in), numpy.sum(D_out)

adjMat = numpy.fromfunction(graphgenerate.adjFun, (50, 50))
(D_in, D_out, d_in, d_out) = degrees(adjMat, 2)
print D_in, D_out, d_in, d_out
