import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.mlab import PCA
from scipy import stats

node1 = [[18, 17, 16, 15], [5, 5, 5, 5]]
node2 = [[10,  9,  8,  7,  6,  5,  4], [1, 1]]

def softmax(x):
  eX = np.exp(x - np.max(x))
  return eX / eX.sum()

x1 = [1, 2, 3, 4, 5]
x2 = [1, 2, 3, 4, 5, 6, 7]


print np.array(np.random.randint(10,size=(10,3)))

#print np.array(node1)
#print PCA(np.array([[1,2,3,4],[3,4,5,6],[1,2,2,2],[3,4,1,2],[4,6,5,5]]))
#quit()
#pcaY1 = PCA(np.array(node1))
#pcaY2 = PCA(np.array(node2))

softmaxY1 = softmax(node1[0])
softmaxY2 = softmax(node1[1])
softmaxY3 = softmax(node2[0])
softmaxY4 = softmax(node2[1])

plt.style.use("seaborn-whitegrid")
plt.figure()

plt.subplot(211)
plt.plot(np.arange(0, len(node1[0])), softmaxY1, 'o', markersize=4)
plt.plot(np.arange(0, len(node1[1])), softmaxY2, 'o', markersize=4)

plt.subplot(212)
plt.plot(np.arange(0, len(node2[0])), softmaxY3, 'o', markersize=4)
plt.plot(np.arange(0, len(node2[1])), softmaxY4, 'o', markersize=4)

#print stats.kstest(node1[0], 'norm')
print ""
print ""
print "if statistic is small or pvalue high, then F(x)=G(x)"
print "ks test for decreasing distri"
print softmaxY1, softmaxY3
print stats.ks_2samp(softmaxY1, softmaxY3)
print "ks test for stable distri"
print softmaxY2, softmaxY4
print stats.ks_2samp(softmaxY2, softmaxY4)

plt.show()
