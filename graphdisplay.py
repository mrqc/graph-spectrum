import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import motifs
import numpy as np

def renderDiGraphFromAdj(adjMat):
  graph = nx.DiGraph(adjMat)
  #pos = nx.circular_layout(graph)
  #pos = nx.spectral_layout(graph)
  pos = nx.shell_layout(graph)
  #nx.draw(graph, cmap = plt.get_cmap('jet'))
  nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes, ax=None, node_size=400)
  nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=1.0, alpha=0.5)
  labels = {}
  for index in range(0, len(adjMat)):
    labels[index] = str(index)
  nx.draw_networkx_labels(graph, pos, labels)

def renderUndirGraphFromAdj(adjMat):
  graph = nx.Graph(adjMat)
  #pos = nx.spring_layout(graph, k=400)
  #pos = nx.shell_layout(graph)
  #pos = nx.spectral_layout(graph)
  pos = nx.circular_layout(graph)
  #nx.draw(graph, cmap = plt.get_cmap('jet'))
  #nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes, node_size=np.empty(len(graph.nodes)).fill(50))
  nx.draw_networkx_nodes(graph, pos, nodelist=graph.nodes, node_size=400)
  nx.draw_networkx_edges(graph, pos, edgelist=graph.edges, width=1.0, alpha=0.5)
  labels = {}
  for index in range(0, len(adjMat)):
    labels[index] = str(index)
  nx.draw_networkx_labels(graph, pos, labels)
