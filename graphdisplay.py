import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import motifs

def renderGraphFromAdj(adjMat):
  graph = nx.DiGraph(adjMat)
  nx.draw(graph, cmap = plt.get_cmap('jet'))
  plt.show()
