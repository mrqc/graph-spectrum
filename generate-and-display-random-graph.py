import numpy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import motifs
import graphgenerate
import graphdisplay

adjMat = numpy.fromfunction(graphgenerate.adjFun, (50, 50))
graphdisplay.renderGraphFromAdj(adjMat)
