import numpy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def adjFun(x, y):
  ret = numpy.empty((x.shape[0], y.shape[1]))
  for xIndex in range(0, x.shape[0]):
    for yIndex in range(0, y.shape[1]):
      val = numpy.random.normal()
      if val >= 0.5:
        ret[xIndex][yIndex] = 1
      else:
        ret[xIndex][yIndex] = 0
  return ret
adjMat = numpy.fromfunction(adjFun, (10, 10))
graph = nx.DiGraph(adjMat)
nx.draw(graph, cmap = plt.get_cmap('jet'))
plt.show()
